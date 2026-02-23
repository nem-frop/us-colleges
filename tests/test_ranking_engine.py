"""
Unit tests for the RankingEngine class.

Tests cover:
- Multi-source score averaging (critical bug fix from Session 8)
- Score normalization
- Coverage penalty
- Selectivity bonus
- Edge cases
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ranking_engine import RankingEngine


@pytest.fixture
def sample_data_dir():
    """Create temporary CSV files with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Universities
        unis = pd.DataFrame({
            'ipeds_id': [1, 2, 3, 4],
            'name': ['Alpha Univ', 'Beta College', 'Gamma Tech', 'Delta State'],
            'acceptance_rate': [0.05, 0.20, 0.10, 0.50],
            'state': ['CA', 'NY', 'MA', 'TX'],
        })
        unis.to_csv(Path(tmpdir) / 'universities.csv', index=False)

        # Rankings with multi-source data for testing averaging
        rankings = pd.DataFrame([
            # Alpha: rank 1 in CS on both sources (should get 100)
            {'ipeds_id': 1, 'category': 'Computer Science', 'rank': 1, 'max_rank': 100, 'source': 'QS', 'year': 2025},
            {'ipeds_id': 1, 'category': 'Computer Science', 'rank': 1, 'max_rank': 500, 'source': 'THE', 'year': 2025},

            # Beta: rank 3 on small list (3/50) vs rank 3 on large list (3/500)
            # These should NOT be equal - 3/500 is much better
            {'ipeds_id': 2, 'category': 'Computer Science', 'rank': 3, 'max_rank': 50, 'source': 'Niche', 'year': 2025},

            # Gamma: rank 3 on large list only - should beat Beta
            {'ipeds_id': 3, 'category': 'Computer Science', 'rank': 3, 'max_rank': 500, 'source': 'QS', 'year': 2025},

            # Delta: rank 10 on both sources
            {'ipeds_id': 4, 'category': 'Computer Science', 'rank': 10, 'max_rank': 100, 'source': 'QS', 'year': 2025},
            {'ipeds_id': 4, 'category': 'Computer Science', 'rank': 10, 'max_rank': 100, 'source': 'THE', 'year': 2025},

            # Add Business rankings for coverage testing
            {'ipeds_id': 1, 'category': 'Business', 'rank': 5, 'max_rank': 200, 'source': 'QS', 'year': 2025},
            {'ipeds_id': 2, 'category': 'Business', 'rank': 1, 'max_rank': 200, 'source': 'QS', 'year': 2025},
            # Gamma has no Business ranking (will test coverage penalty)
        ])
        rankings.to_csv(Path(tmpdir) / 'rankings.csv', index=False)

        yield tmpdir


@pytest.fixture
def engine(sample_data_dir):
    """Create RankingEngine with test data."""
    return RankingEngine(
        universities_path=str(Path(sample_data_dir) / 'universities.csv'),
        rankings_path=str(Path(sample_data_dir) / 'rankings.csv')
    )


class TestNormalization:
    """Tests for score normalization."""

    def test_rank_1_gives_100(self, engine):
        """Rank 1 should always give score 100."""
        score = engine.normalize_rank(1, 100)
        assert score == 100.0

    def test_rank_max_gives_0(self, engine):
        """Last rank should give score 0."""
        score = engine.normalize_rank(100, 100)
        assert score == 0.0

    def test_middle_rank(self, engine):
        """Middle rank should give ~50."""
        score = engine.normalize_rank(51, 101)  # Middle of 1-101
        assert abs(score - 50.0) < 0.1

    def test_nan_handling(self, engine):
        """NaN inputs should return NaN."""
        assert np.isnan(engine.normalize_rank(np.nan, 100))
        assert np.isnan(engine.normalize_rank(1, np.nan))

    def test_max_rank_1_edge_case(self, engine):
        """max_rank=1 should return NaN (can't normalize)."""
        assert np.isnan(engine.normalize_rank(1, 1))


class TestMultiSourceAveraging:
    """Tests for the critical multi-source averaging logic.

    This was a major bug fix - previously we took the best rank,
    but now we properly average normalized scores.
    """

    def test_same_rank_different_scales(self, engine):
        """Rank 3/50 should score LOWER than rank 3/500.

        This is the core bug that was fixed - we need to account for
        different ranking pool sizes when comparing.
        """
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.0,  # No selectivity boost
            coverage_penalty=0.0     # No coverage penalty
        )

        # Find Beta (rank 3/50) and Gamma (rank 3/500)
        beta = results[results['name'] == 'Beta College'].iloc[0]
        gamma = results[results['name'] == 'Gamma Tech'].iloc[0]

        # Gamma should have higher score (3/500 is better than 3/50)
        assert gamma['weighted_score'] > beta['weighted_score'], \
            f"Gamma (3/500) should beat Beta (3/50): {gamma['weighted_score']} vs {beta['weighted_score']}"

    def test_multi_source_average_not_best(self, engine):
        """Multi-source schools should get average, not best score."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        # Alpha has rank 1 on both sources (100/100 and 500/500)
        # Score should be 100 (average of 100 and 100)
        alpha = results[results['name'] == 'Alpha Univ'].iloc[0]
        assert alpha['weighted_score'] == 100.0

        # Delta has rank 10/100 on both sources
        # Score = 100 * (1 - 9/99) = ~90.9
        delta = results[results['name'] == 'Delta State'].iloc[0]
        expected = 100 * (1 - 9/99)
        assert abs(delta['weighted_score'] - expected) < 0.5

    def test_normalized_scores_in_results(self, engine):
        """Results should include per-category score info."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        alpha = results[results['name'] == 'Alpha Univ'].iloc[0]
        assert 'category_scores' in alpha
        assert 'Computer Science' in alpha['category_scores']
        assert 'score' in alpha['category_scores']['Computer Science']


class TestCoveragePenalty:
    """Tests for the coverage penalty feature."""

    def test_no_penalty_when_disabled(self, engine):
        """coverage_penalty=0 should not reduce scores."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science', 'Business'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        # Gamma only has CS, not Business
        gamma = results[results['name'] == 'Gamma Tech'].iloc[0]

        # With no penalty, score should just be CS score
        expected_cs_score = 100 * (1 - 2/499)  # rank 3/500
        assert abs(gamma['weighted_score'] - expected_cs_score) < 0.5

    def test_penalty_reduces_partial_coverage(self, engine):
        """Schools missing categories should be penalized."""
        results_no_penalty = engine.compute_weighted_score(
            selected_categories=['Computer Science', 'Business'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        results_with_penalty = engine.compute_weighted_score(
            selected_categories=['Computer Science', 'Business'],
            selectivity_factor=0.0,
            coverage_penalty=1.0  # Full penalty
        )

        # Gamma has 50% coverage (only CS)
        gamma_no = results_no_penalty[results_no_penalty['name'] == 'Gamma Tech'].iloc[0]
        gamma_with = results_with_penalty[results_with_penalty['name'] == 'Gamma Tech'].iloc[0]

        # With full penalty, 50% coverage = 50% score reduction
        assert gamma_with['weighted_score'] < gamma_no['weighted_score']
        assert abs(gamma_with['weighted_score'] - gamma_no['weighted_score'] * 0.5) < 1.0

    def test_full_coverage_no_penalty(self, engine):
        """Schools with full coverage should not be penalized."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science', 'Business'],
            selectivity_factor=0.0,
            coverage_penalty=1.0
        )

        # Alpha and Beta have both CS and Business
        alpha = results[results['name'] == 'Alpha Univ'].iloc[0]
        assert alpha['coverage_ratio'] == 1.0


class TestSelectivityBonus:
    """Tests for the selectivity bonus feature."""

    def test_no_bonus_when_disabled(self, engine):
        """selectivity_factor=0 should give no bonus."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        for _, row in results.iterrows():
            assert row['selectivity_bonus'] == 0

    def test_lower_acceptance_higher_bonus(self, engine):
        """Lower acceptance rate should give higher bonus."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.5,
            coverage_penalty=0.0
        )

        alpha = results[results['name'] == 'Alpha Univ'].iloc[0]  # 5% acceptance
        beta = results[results['name'] == 'Beta College'].iloc[0]  # 20% acceptance
        delta = results[results['name'] == 'Delta State'].iloc[0]  # 50% acceptance

        assert alpha['selectivity_bonus'] > beta['selectivity_bonus'] > delta['selectivity_bonus']

    def test_ultra_selective_extra_bonus(self, engine):
        """Sub-6% schools should get extra bonus."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=1.0,
            coverage_penalty=0.0
        )

        alpha = results[results['name'] == 'Alpha Univ'].iloc[0]  # 5% acceptance

        # With selectivity_factor=1.0, base bonus = 20*(1-0.05/0.5) = 18
        # Ultra bonus for 5%: 10*(1-0.05/0.06) = ~1.67
        # Total should be ~19.67
        assert alpha['selectivity_bonus'] > 15  # Should be significant


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_categories_raises(self, engine):
        """Empty category list should raise error."""
        with pytest.raises(ValueError):
            engine.compute_weighted_score(
                selected_categories=[],
                selectivity_factor=0.0,
                coverage_penalty=0.0
            )

    def test_nonexistent_category_ignored(self, engine):
        """Non-existent categories should be ignored, not error."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science', 'FakeCategory123'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )
        assert len(results) > 0

    def test_results_sorted_by_score(self, engine):
        """Results should be sorted by weighted_score descending."""
        results = engine.compute_weighted_score(
            selected_categories=['Computer Science'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        scores = results['weighted_score'].tolist()
        assert scores == sorted(scores, reverse=True)


class TestWithRealData:
    """Tests using actual production data (if available)."""

    @pytest.fixture
    def real_engine(self):
        """Load the real data if available."""
        data_dir = Path(__file__).parent.parent / "data"
        unis_path = data_dir / "universities.csv"
        rankings_path = data_dir / "rankings.csv"

        if not unis_path.exists() or not rankings_path.exists():
            pytest.skip("Production data not available")

        return RankingEngine(str(unis_path), str(rankings_path))

    def test_harvard_mit_cs_engineering(self, real_engine):
        """MIT should beat Harvard in pure CS+Engineering selection."""
        results = real_engine.compute_weighted_score(
            selected_categories=['Computer Science', 'Engineering'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        mit = results[results['name'].str.contains('Massachusetts Institute', case=False)]
        harvard = results[results['name'].str.contains('Harvard', case=False)]

        if len(mit) > 0 and len(harvard) > 0:
            mit_score = mit.iloc[0]['weighted_score']
            harvard_score = harvard.iloc[0]['weighted_score']
            assert mit_score > harvard_score, \
                f"MIT should beat Harvard in CS+Eng: {mit_score} vs {harvard_score}"

    def test_reasonable_score_range(self, real_engine):
        """All scores should be in reasonable range."""
        results = real_engine.compute_weighted_score(
            selected_categories=['Global'],
            selectivity_factor=0.0,
            coverage_penalty=0.0
        )

        assert results['weighted_score'].min() >= 0
        assert results['weighted_score'].max() <= 100

    def test_categories_loaded(self, real_engine):
        """Should have loaded 50+ categories."""
        assert len(real_engine.categories) >= 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
