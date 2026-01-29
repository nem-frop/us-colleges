"""
Generate test outputs for red team validation.

Creates CSV files with rankings for common student profiles that
the consulting team can review for sanity checking.
"""

import pandas as pd
from pathlib import Path
from ranking_engine import RankingEngine

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "test_outputs"


def generate_test_outputs():
    """Generate test ranking outputs for various student profiles."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    engine = RankingEngine(
        universities_path=str(DATA_DIR / "universities.csv"),
        rankings_path=str(DATA_DIR / "rankings.csv")
    )

    # Define test profiles
    test_profiles = {
        # Profile name: (categories, selectivity, coverage_penalty, description)
        "business_economics": {
            "categories": ["Business", "Economics", "Accounting & Finance"],
            "selectivity": 0.3,
            "coverage": 0.5,
            "description": "Business/Economics focus student"
        },
        "stem_cs_engineering": {
            "categories": ["Computer Science", "Engineering", "Mathematics"],
            "selectivity": 0.2,
            "coverage": 0.5,
            "description": "CS/Engineering student"
        },
        "premed": {
            "categories": ["Pre-Med", "Biology", "Chemistry", "Medicine"],
            "selectivity": 0.3,
            "coverage": 0.5,
            "description": "Pre-med student"
        },
        "humanities": {
            "categories": ["English", "History", "Philosophy", "Psychology"],
            "selectivity": 0.2,
            "coverage": 0.5,
            "description": "Humanities-focused student"
        },
        "liberal_arts_overall": {
            "categories": ["Liberal Arts", "USA Overall", "Global"],
            "selectivity": 0.3,
            "coverage": 0.3,
            "description": "Overall liberal arts rankings"
        },
        "all_categories_balanced": {
            "categories": list(engine.categories),
            "selectivity": 0.2,
            "coverage": 0.5,
            "description": "All categories - balanced view"
        },
        "all_categories_high_selectivity": {
            "categories": list(engine.categories),
            "selectivity": 1.0,
            "coverage": 0.5,
            "description": "All categories - max selectivity boost"
        },
        "all_categories_low_coverage_penalty": {
            "categories": list(engine.categories),
            "selectivity": 0.2,
            "coverage": 0.0,
            "description": "All categories - no coverage penalty (specialist schools can rank high)"
        },
        "all_categories_high_coverage_penalty": {
            "categories": list(engine.categories),
            "selectivity": 0.2,
            "coverage": 1.0,
            "description": "All categories - max coverage penalty (need rankings in most subjects)"
        },
        "art_design": {
            "categories": ["Art & Design", "Architecture", "Film & Media"],
            "selectivity": 0.2,
            "coverage": 0.3,
            "description": "Art/Design student"
        },
        "social_sciences": {
            "categories": ["Psychology", "Political Science", "Sociology", "Anthropology", "Education"],
            "selectivity": 0.2,
            "coverage": 0.5,
            "description": "Social sciences student"
        },
    }

    # Generate outputs
    summary_rows = []

    for profile_name, config in test_profiles.items():
        print(f"\nGenerating: {profile_name}")
        print(f"  Categories: {config['categories']}")

        results = engine.compute_weighted_score(
            selected_categories=config['categories'],
            selectivity_factor=config['selectivity'],
            coverage_penalty=config['coverage']
        )

        # Select columns for output
        output_cols = [
            'weighted_rank', 'name', 'weighted_score', 'base_score',
            'coverage_ratio', 'num_categories', 'selectivity_bonus',
            'acceptance_rate', 'state', 'public_private',
            'undergrad_enrollment', 'tuition_fees', 'sat_25_overall', 'sat_75_overall'
        ]
        available_cols = [c for c in output_cols if c in results.columns]
        output_df = results[available_cols].head(100)

        # Save to CSV
        output_file = OUTPUT_DIR / f"{profile_name}_top100.csv"
        output_df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")

        # Add to summary
        top5 = results.head(5)['name'].tolist()
        summary_rows.append({
            'Profile': profile_name,
            'Description': config['description'],
            'Categories': ', '.join(config['categories'][:5]) + ('...' if len(config['categories']) > 5 else ''),
            'Selectivity': config['selectivity'],
            'Coverage Penalty': config['coverage'],
            'Total Schools': len(results),
            '#1': top5[0] if len(top5) > 0 else '',
            '#2': top5[1] if len(top5) > 1 else '',
            '#3': top5[2] if len(top5) > 2 else '',
            '#4': top5[3] if len(top5) > 3 else '',
            '#5': top5[4] if len(top5) > 4 else '',
        })

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "00_SUMMARY.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\n\nSummary saved: {summary_file}")

    # Also create a readable summary
    readme_content = """# Test Outputs for Red Team Validation

These CSV files contain ranking outputs for various student profiles.
Use them to sanity-check the algorithm against your consulting experience.

## Files

| File | Description |
|------|-------------|
"""
    for profile_name, config in test_profiles.items():
        readme_content += f"| `{profile_name}_top100.csv` | {config['description']} |\n"

    readme_content += """
## Parameters Explained

- **Selectivity**: 0-1 scale. How much to boost selective schools (low acceptance rate).
  - 0 = ignore selectivity
  - 1 = heavily favor selective schools

- **Coverage Penalty**: 0-1 scale. How much to penalize schools missing rankings.
  - 0 = no penalty (specialist schools like RISD can rank high if they excel in their niche)
  - 1 = strong penalty (need rankings in most selected subjects to rank well)

## Key Questions for Validation

1. Do the top 10-20 schools in each profile look reasonable?
2. Are there any surprising entries that seem wrong?
3. For "all_categories" tests, does high coverage penalty correctly filter out specialist schools?
4. Does the selectivity boost work as expected (more selective schools rise)?

## Known Behaviors

- Schools only ranked in 1-2 categories will rank lower when coverage penalty > 0
- Very selective schools (< 10% acceptance) get a bonus when selectivity > 0
- Large public research universities may rank highly if coverage penalty is low

Generated by: generate_test_outputs.py
"""

    readme_file = OUTPUT_DIR / "README.md"
    readme_file.write_text(readme_content)
    print(f"README saved: {readme_file}")

    print("\n" + "="*60)
    print("TEST OUTPUT GENERATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files generated: {len(test_profiles) + 2}")  # +2 for summary and readme


if __name__ == "__main__":
    generate_test_outputs()
