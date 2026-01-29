"""
Weighted Ranking Engine for College Data

This module provides a clean, mathematically sound approach to computing
weighted college rankings based on user-selected subject preferences.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


class RankingEngine:
    """
    Computes weighted rankings for universities based on selected criteria.

    The algorithm:
    1. For each selected category, compute a normalized score (0-100, higher = better)
    2. Apply optional weighting to categories
    3. Aggregate scores using a configurable method
    4. Optionally factor in selectivity (acceptance rate)
    """

    def __init__(self, universities_path: str, rankings_path: str):
        """Load data from CSV files."""
        self.universities = pd.read_csv(universities_path)
        self.rankings = pd.read_csv(rankings_path)

        # Pre-compute category list
        self.categories = sorted(self.rankings['category'].unique())

        # Create a pivot table for faster lookups: ipeds_id x category -> (rank, max_rank)
        self._build_rank_matrix()

    def _build_rank_matrix(self):
        """Build efficient lookup structures for rankings."""
        # For each university and category, store the best rank
        # (some universities have multiple ranks in same category from different sources)
        best_ranks = self.rankings.groupby(['ipeds_id', 'category']).agg({
            'rank': 'min',  # Best (lowest) rank
            'max_rank': 'max'  # Use max of the max_ranks for normalization
        }).reset_index()

        # Pivot to wide format
        self.rank_matrix = best_ranks.pivot(
            index='ipeds_id',
            columns='category',
            values='rank'
        )

        self.max_rank_matrix = best_ranks.pivot(
            index='ipeds_id',
            columns='category',
            values='max_rank'
        )

    def normalize_rank(self, rank: float, max_rank: float) -> float:
        """
        Convert a rank to a 0-100 score where 100 = best (rank 1).

        Formula: score = 100 * (1 - (rank - 1) / (max_rank - 1))

        This gives:
        - Rank 1 → 100
        - Rank max_rank → 0
        - Linear interpolation between
        """
        if pd.isna(rank) or pd.isna(max_rank) or max_rank <= 1:
            return np.nan
        return 100 * (1 - (rank - 1) / (max_rank - 1))

    def compute_weighted_score(
        self,
        selected_categories: List[str],
        category_weights: Optional[Dict[str, float]] = None,
        selectivity_factor: float = 0.0,
        coverage_penalty: float = 0.5
    ) -> pd.DataFrame:
        """
        Compute weighted ranking scores for all universities.

        Args:
            selected_categories: List of category names to include
            category_weights: Optional dict of category -> weight (default: equal weights)
            selectivity_factor: How much to weight acceptance rate (0 = ignore, 1 = heavy weight)
            coverage_penalty: How much to penalize schools missing rankings (0 = no penalty, 1 = full penalty)
                             This prevents schools ranked in only 1-2 categories from dominating.

        Returns:
            DataFrame with universities and their weighted scores
        """
        if not selected_categories:
            raise ValueError("Must select at least one category")

        # Default to equal weights
        if category_weights is None:
            category_weights = {cat: 1.0 for cat in selected_categories}

        # Normalize weights to sum to 1
        total_weight = sum(category_weights.get(cat, 1.0) for cat in selected_categories)
        norm_weights = {cat: category_weights.get(cat, 1.0) / total_weight
                        for cat in selected_categories}

        # Count how many selected categories actually exist in our data
        valid_categories = [cat for cat in selected_categories if cat in self.rank_matrix.columns]
        num_valid_categories = len(valid_categories)

        # Compute normalized scores for each category
        results = []
        for ipeds_id in self.rank_matrix.index:
            scores = []
            weights = []
            category_scores = {}

            for cat in selected_categories:
                if cat not in self.rank_matrix.columns:
                    continue

                rank = self.rank_matrix.loc[ipeds_id, cat]
                max_rank = self.max_rank_matrix.loc[ipeds_id, cat]

                if pd.notna(rank):
                    score = self.normalize_rank(rank, max_rank)
                    scores.append(score)
                    weights.append(norm_weights[cat])
                    category_scores[cat] = {
                        'rank': int(rank),
                        'max_rank': int(max_rank) if pd.notna(max_rank) else None,
                        'score': round(score, 1)
                    }

            # Skip if no scores at all
            if len(scores) == 0:
                continue

            # Compute base score (weighted mean of available scores)
            base_score = np.average(scores, weights=weights)

            # Apply coverage penalty: schools missing many rankings get penalized
            # coverage_ratio = what fraction of selected categories does this school have?
            coverage_ratio = len(scores) / num_valid_categories if num_valid_categories > 0 else 1

            # Penalty formula: score * (1 - penalty * (1 - coverage))
            # If coverage = 100%, no penalty. If coverage = 10%, significant penalty.
            # With coverage_penalty=0.5 and coverage=20%, multiplier = 1 - 0.5*0.8 = 0.6 (40% reduction)
            coverage_multiplier = 1 - coverage_penalty * (1 - coverage_ratio)
            adjusted_score = base_score * coverage_multiplier

            results.append({
                'ipeds_id': ipeds_id,
                'base_score': base_score,
                'coverage_ratio': coverage_ratio,
                'adjusted_score': adjusted_score,
                'num_categories': len(scores),
                'category_scores': category_scores
            })

        # Convert to DataFrame
        df = pd.DataFrame(results)

        if df.empty:
            return df

        # Merge with university info
        df = df.merge(self.universities, on='ipeds_id', how='left')

        # Apply selectivity adjustment if requested
        if selectivity_factor > 0 and 'acceptance_rate' in df.columns:
            # Lower acceptance rate = more selective = higher bonus
            # Using exponential scaling so ultra-selective schools (sub-6%) stand out more
            # Two-tier approach: extra bonus for sub-6% schools
            def calc_selectivity_bonus(ar):
                if pd.isna(ar):
                    return 0
                ar_capped = min(ar, 0.5)  # Cap at 50%
                # Base bonus: linear scale (same as before)
                base_bonus = 20 * (1 - ar_capped / 0.5)
                # Extra bonus for ultra-selective (sub-6%): up to 10 extra points
                ultra_bonus = 0
                if ar_capped < 0.06:
                    # Linear scale from 6% (0 extra) to 3% (10 extra)
                    ultra_bonus = 10 * (1 - ar_capped / 0.06)
                return selectivity_factor * (base_bonus + ultra_bonus)

            df['selectivity_bonus'] = df['acceptance_rate'].apply(calc_selectivity_bonus)
            df['weighted_score'] = df['adjusted_score'] + df['selectivity_bonus']
        else:
            df['selectivity_bonus'] = 0
            df['weighted_score'] = df['adjusted_score']

        # Compute final rank
        df['weighted_rank'] = df['weighted_score'].rank(ascending=False, method='min').astype(int)

        # Sort by weighted score descending
        df = df.sort_values('weighted_score', ascending=False).reset_index(drop=True)

        return df

    def get_university_profile(self, ipeds_id: int) -> Dict:
        """Get detailed profile for a single university."""
        uni = self.universities[self.universities['ipeds_id'] == ipeds_id]
        if uni.empty:
            return None

        uni_info = uni.iloc[0].to_dict()

        # Get all rankings for this university
        ranks = self.rankings[self.rankings['ipeds_id'] == ipeds_id]
        uni_info['rankings'] = ranks.to_dict('records')

        return uni_info


def demo():
    """Demo the ranking engine."""
    DATA_DIR = Path(__file__).parent.parent / "data"

    engine = RankingEngine(
        universities_path=str(DATA_DIR / "universities.csv"),
        rankings_path=str(DATA_DIR / "rankings.csv")
    )

    print("="*60)
    print("COLLEGE RANKING ENGINE DEMO")
    print("="*60)

    # Example 1: Business + Economics student
    print("\n--- Example 1: Business + Economics Focus ---")
    results = engine.compute_weighted_score(
        selected_categories=['Business', 'Economics', 'Accounting & Finance'],
        selectivity_factor=0.3,
        min_categories=2
    )
    print(results[['weighted_rank', 'name', 'weighted_score', 'acceptance_rate', 'num_categories']].head(15).to_string(index=False))

    # Example 2: STEM student
    print("\n--- Example 2: STEM Focus (CS + Engineering + Math) ---")
    results = engine.compute_weighted_score(
        selected_categories=['Computer Science', 'Engineering', 'Mathematics', 'Physics'],
        selectivity_factor=0.2,
        min_categories=2
    )
    print(results[['weighted_rank', 'name', 'weighted_score', 'acceptance_rate', 'num_categories']].head(15).to_string(index=False))

    # Example 3: Pre-Med
    print("\n--- Example 3: Pre-Med Focus ---")
    results = engine.compute_weighted_score(
        selected_categories=['Pre Medicine', 'Biology', 'Chemistry', 'Biological Sciences'],
        selectivity_factor=0.3,
        min_categories=2
    )
    print(results[['weighted_rank', 'name', 'weighted_score', 'acceptance_rate', 'num_categories']].head(15).to_string(index=False))

    print("\n" + "="*60)
    print(f"Available categories ({len(engine.categories)}):")
    print("="*60)
    for i, cat in enumerate(engine.categories, 1):
        print(f"  {i:2}. {cat}")


if __name__ == "__main__":
    demo()
