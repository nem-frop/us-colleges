"""
Extract and clean college data from the Excel workbook into normalized CSV files.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_FILE = PROJECT_ROOT / "College Rankings Longlist - v9 (Mar _24) - Excel Only.xlsx"


def extract_universities(xl: pd.ExcelFile) -> pd.DataFrame:
    """Extract and clean university institutional data - ALL columns."""

    # Read raw data, skipping header rows
    df = pd.read_excel(xl, sheet_name='Raw Universities Data', header=None)

    # Row 1 contains column names
    headers = df.iloc[1].values

    # Get the actual data starting from row 4 (0-indexed: row 4 is first data row)
    data = df.iloc[4:].copy()
    data.columns = headers
    data = data.reset_index(drop=True)

    # Column name mapping (original -> clean name)
    column_mapping = {
        'IPEDS_ID': 'ipeds_id',
        'Institution': 'name',
        'Admissions Website': 'website',
        'Mission Statement': 'mission_statement',
        'Description of University': 'description',
        'Popular Domains': 'popular_domains',
        'All Domains': 'all_domains',
        'Uses Common App?': 'uses_common_app',
        'Total Applicants #': 'total_applicants',
        'Total Acceptance Rate %': 'acceptance_rate',
        'Total Yield (% Admitted who enroll)': 'yield_rate',
        'Male Applicants #': 'male_applicants',
        'Acceptance Rate Male %': 'acceptance_rate_male',
        'Female Applicants #': 'female_applicants',
        'Acceptance Rate Female %': 'acceptance_rate_female',
        'Additional Admission Plans': 'admission_plans',
        'Early Decision (ED) Acceptance Rate (1)': 'ed_acceptance_rate',
        'ED to RD         Acceptance Ratio': 'ed_to_rd_ratio',
        'ED Admits as Percent of Freshman Class': 'ed_pct_freshman',
        'General/A&S Selectivity Zone': 'selectivity_zone_general',
        'CS/Eng. Selectivity Zone': 'selectivity_zone_cs_eng',
        'Econ/Biz. Selectivity Zone': 'selectivity_zone_econ_biz',
        'Premed/Bio Selectivity Zone': 'selectivity_zone_premed',
        'Recommendations Required?': 'recommendations_required',
        'Admissions Test Scores Required?': 'test_scores_required',
        'TOEFL Required?': 'toefl_required',
        'SAT 25%ile Overall': 'sat_25_overall',
        'SAT 75%ile Overall': 'sat_75_overall',
        'SAT English 25%ile': 'sat_english_25',
        'SAT English 75%ile': 'sat_english_75',
        'SAT Math 25%ile': 'sat_math_25',
        'SAT Math 75%ile': 'sat_math_75',
        'ACT Composite 25%ile': 'act_25',
        'ACT Composite 75%ile': 'act_75',
        'ACT English 25%ile': 'act_english_25',
        'ACT English 75%ile': 'act_english_75',
        'ACT Math 25%ile': 'act_math_25',
        'ACT Math 75%ile': 'act_math_75',
        'Undergraduate Enrollment': 'undergrad_enrollment',
        'Graduate Enrollment': 'grad_enrollment',
        'Undergraduate Ethnic White %': 'pct_white',
        'Undergraduate International Student %': 'pct_international',
        'Student to Faculty Ratio': 'student_faculty_ratio',
        '% Students Registered with Disabilities Office': 'pct_disabilities',
        'Special Characteristics (Coed, HBCU)': 'special_characteristics',
        'Scholarship Recipients #': 'scholarship_recipients',
        'Scholarship Recipients %': 'pct_scholarship',
        'Average Scholarship Amount': 'avg_scholarship',
        'Campus Setting': 'campus_setting',
        'University Type': 'university_type',
        'Public/Private University': 'public_private',
        '% of In-State Students (Top Public Schools)': 'pct_in_state',
        'Religious Affiliation': 'religious_affiliation',
        'City': 'city',
        'State': 'state',
        'Region': 'region',
        'Tuition and Fees (USD)': 'tuition_fees',
        'Room and Board (USD)': 'room_board',
        'Estimated Expenses Total (USD)': 'total_expenses',
        'Salary after 10 years (USD)': 'salary_10yr',
        ' # Singaporean Students Enrolled': 'students_singapore',
        '# Vietnamese Students Enrolled': 'students_vietnam',
        '# Malaysian Students Enrolled': 'students_malaysia',
        '# Indonesian Students Enrolled': 'students_indonesia',
        '# Thai Students Enrolled': 'students_thailand',
        'Highest Sport Division': 'sport_division',
        'Total Revenue - All Sports': 'sports_revenue',
        'Total Expenses - All Sports': 'sports_expenses',
        'Total Athletes': 'total_athletes',
        '% Student Athletes/Undergraduates+Graduates': 'pct_athletes',
        '# Female Athletes': 'female_athletes',
        '# Male Athletes': 'male_athletes',
    }

    # Find which columns exist and map them
    available_cols = {}
    for orig, new in column_mapping.items():
        # Try exact match first
        if orig in data.columns:
            available_cols[orig] = new
        else:
            # Try to find similar column
            for col in data.columns:
                if isinstance(col, str) and isinstance(orig, str):
                    if orig.lower().replace(' ', '') == col.lower().replace(' ', ''):
                        available_cols[col] = new
                        break

    # Select and rename columns
    cols_to_keep = [c for c in data.columns if c in available_cols]
    clean_df = data[cols_to_keep].copy()
    clean_df.columns = [available_cols[c] for c in clean_df.columns]

    # Clean data types
    if 'ipeds_id' in clean_df.columns:
        clean_df['ipeds_id'] = pd.to_numeric(clean_df['ipeds_id'], errors='coerce').astype('Int64')

    # Numeric columns
    numeric_cols = [
        'total_applicants', 'acceptance_rate', 'yield_rate',
        'male_applicants', 'acceptance_rate_male', 'female_applicants', 'acceptance_rate_female',
        'ed_acceptance_rate', 'ed_to_rd_ratio', 'ed_pct_freshman',
        'sat_25_overall', 'sat_75_overall', 'sat_english_25', 'sat_english_75',
        'sat_math_25', 'sat_math_75', 'act_25', 'act_75',
        'act_english_25', 'act_english_75', 'act_math_25', 'act_math_75',
        'undergrad_enrollment', 'grad_enrollment', 'pct_white', 'pct_international',
        'student_faculty_ratio', 'pct_disabilities', 'scholarship_recipients',
        'pct_scholarship', 'avg_scholarship', 'pct_in_state',
        'tuition_fees', 'room_board', 'total_expenses', 'salary_10yr',
        'students_singapore', 'students_vietnam', 'students_malaysia',
        'students_indonesia', 'students_thailand',
        'sports_revenue', 'sports_expenses', 'total_athletes', 'pct_athletes',
        'female_athletes', 'male_athletes'
    ]

    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')

    # Remove rows without IPEDS ID
    clean_df = clean_df.dropna(subset=['ipeds_id'])

    print(f"Extracted {len(clean_df)} universities with {len(clean_df.columns)} columns")
    return clean_df


def extract_rankings(xl: pd.ExcelFile) -> pd.DataFrame:
    """Extract rankings into a normalized long format with consolidated categories."""

    # Read raw rankings data
    df = pd.read_excel(xl, sheet_name='Raw Rankings Data', header=None)

    # Parse header structure:
    # Row 0: Category names (AA_Global Rank, Accounting...Finance, etc.)
    # Row 1: Source (USNews, Times, QS, Niche, etc.)
    # Row 2: Year
    # Row 3: Full column ID
    # Row 4: Maximum rank value
    # Data starts at row 5

    categories = df.iloc[0, 2:].values  # Skip first two columns
    sources = df.iloc[1, 2:].values
    years = df.iloc[2, 2:].values
    max_ranks = df.iloc[4, 2:].values

    # Get IPEDS IDs and ranking values
    data = df.iloc[5:].copy()
    ipeds_ids = data.iloc[:, 0].values
    rank_values = data.iloc[:, 2:].values

    # Build long-format dataframe
    records = []
    for i, ipeds_id in enumerate(ipeds_ids):
        if pd.isna(ipeds_id):
            continue
        for j in range(len(categories)):
            rank_val = rank_values[i, j]
            if pd.notna(rank_val) and rank_val != '' and rank_val != 0:
                try:
                    rank_num = float(rank_val)
                    max_rank = float(max_ranks[j]) if pd.notna(max_ranks[j]) else None

                    # Clean up category name
                    cat = str(categories[j]) if pd.notna(categories[j]) else ''
                    cat = cat.replace('...', ' & ').replace('.', ' ').strip()

                    records.append({
                        'ipeds_id': int(float(ipeds_id)),
                        'category_raw': cat,
                        'source': str(sources[j]) if pd.notna(sources[j]) else 'Unknown',
                        'year': int(float(years[j])) if pd.notna(years[j]) else None,
                        'rank': int(rank_num),
                        'max_rank': int(max_rank) if max_rank else None,
                    })
                except (ValueError, TypeError):
                    continue

    rankings_df = pd.DataFrame(records)

    # Consolidate duplicate/similar categories into unified names
    # This maps similar rankings to a single category for easier use
    category_consolidation = {
        # Global/Overall rankings
        'AA_Global Rank': 'Global',
        'AA_USA Rank': 'USA Overall',
        'AA_Liberal Arts': 'Liberal Arts',

        # Business & Economics
        'Accounting': 'Accounting & Finance',
        'Accounting & Finance': 'Accounting & Finance',
        'Finance': 'Accounting & Finance',
        'Business': 'Business',
        'Business & Management Studies': 'Business',
        'Economics': 'Economics',
        'Economics & Econometrics': 'Economics',

        # Computer Science & Engineering
        'Computer Science': 'Computer Science',
        'Computer Science & Information Systems': 'Computer Science',
        'Information Technology': 'Computer Science',
        'Engineering': 'Engineering',
        'Engineering & Technology': 'Engineering',
        'Engineering & Chemical': 'Engineering (Chemical)',
        'Engineering & Civil & Structural': 'Engineering (Civil)',
        'Engineering & Electrical & Electronic': 'Engineering (Electrical)',
        'Engineering & Mechanical  Aeronautical & Manufacturing': 'Engineering (Mechanical)',
        'Engineering & Mineral & Mining': 'Engineering (Mining)',

        # Sciences
        'Biology': 'Biology',
        'Biological Sciences': 'Biology',
        'Life Sciences & Medicine': 'Life Sciences & Medicine',
        'Chemistry': 'Chemistry',
        'Physics': 'Physics',
        'Physics & Astronomy': 'Physics',
        'Mathematics': 'Mathematics',
        'Statistics & Operational Research': 'Mathematics',
        'Natural Sciences': 'Natural Sciences',
        'Environmental Science': 'Environmental Science',
        'Environmental Sciences': 'Environmental Science',
        'Earth & Marine Sciences': 'Earth Sciences',
        'Geology': 'Earth Sciences',
        'Geophysics': 'Earth Sciences',
        'Geography': 'Geography',

        # Health & Medicine
        'Pre Medicine': 'Pre-Med',
        'Medicine': 'Medicine',
        'Life Sciences & Medicine': 'Medicine',  # Consolidated with Medicine
        'Nursing': 'Nursing',
        'Pharmacy & Pharmacology': 'Pharmacy',
        'Public Health': 'Public Health',
        'Dentistry': 'Dentistry',
        'Anatomy & Physiology': 'Anatomy & Physiology',
        'Veterinary Science': 'Veterinary Science',

        # Humanities
        'English': 'English',
        'English Language & Literature': 'English',
        'History': 'History',
        'Philosophy': 'Philosophy',
        'Classics & Ancient History': 'Classics',
        'Modern Languages': 'Languages',  # Consolidated
        'Linguistics': 'Languages',  # Consolidated with Modern Languages
        'Theology  Divinity & Religious Studies': 'Religious Studies',

        # Social Sciences
        'Psychology': 'Psychology',
        'Political Science': 'Political Science',
        'Politics & International Studies': 'Political Science',
        'International Relations': 'Political Science',
        'Sociology': 'Sociology',
        'Anthropology': 'Anthropology',
        'Social Sciences & Management': 'Social Sciences',
        'Social Policy & Administration': 'Public Policy',
        'Public Policy': 'Public Policy',
        'Education': 'Education',
        'Law': 'Law',
        'Development Studies': 'Development Studies',

        # Arts & Communications
        'Art': 'Art & Design',
        'Art & Design': 'Art & Design',
        'Design': 'Art & Design',
        'Architecture': 'Architecture',
        'Architecture & Built Environment': 'Architecture',
        'Performing Arts': 'Performing Arts',
        'Film': 'Film & Media',
        'Communications': 'Film & Media',
        'Communication & Media Studies': 'Film & Media',

        # Other
        'Agriculture & Forestry': 'Agriculture',
        'Hospitality & Leisure Management': 'Hospitality',
        'Sports Management': 'Sports',
        'Sports related Subjects': 'Sports',
        'Library & Information Management': 'Library Science',
        'Materials Science': 'Materials Science',
        'Petroleum Engineering': 'Engineering (Petroleum)',
        'Archaeology': 'Archaeology',
        'Arts & Humanities': 'Arts & Humanities',
    }

    # Apply consolidation
    rankings_df['category'] = rankings_df['category_raw'].map(
        lambda x: category_consolidation.get(x, x)
    )

    # For each university + consolidated category, keep the best rank
    # (since we may have merged multiple sources)
    consolidated = rankings_df.groupby(['ipeds_id', 'category']).agg({
        'rank': 'min',  # Best rank
        'max_rank': 'max',  # Largest scale for normalization
        'source': lambda x: ', '.join(sorted(set(x))),  # Track sources
        'year': 'max',  # Most recent year
    }).reset_index()

    print(f"Extracted {len(rankings_df)} raw records -> {len(consolidated)} consolidated records")
    print(f"Categories: {rankings_df['category_raw'].nunique()} raw -> {consolidated['category'].nunique()} consolidated")
    return consolidated


def get_unique_categories(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """Get list of unique ranking categories with metadata."""

    cats = rankings_df.groupby('category').agg({
        'source': lambda x: ', '.join(sorted(set(', '.join(x).split(', ')))),
        'year': 'max',
        'max_rank': 'max',
        'ipeds_id': 'count'
    }).reset_index()

    cats.columns = ['category', 'sources', 'latest_year', 'max_rank', 'num_schools']
    cats = cats.sort_values('num_schools', ascending=False)

    return cats


def main():
    """Main extraction pipeline."""

    print(f"Loading data from: {SOURCE_FILE}")
    xl = pd.ExcelFile(SOURCE_FILE)

    # Create output directory
    DATA_DIR.mkdir(exist_ok=True)

    # Extract universities
    print("\n--- Extracting Universities ---")
    universities = extract_universities(xl)
    universities.to_csv(DATA_DIR / "universities.csv", index=False)
    print(f"Saved to: {DATA_DIR / 'universities.csv'}")

    # Extract rankings
    print("\n--- Extracting Rankings ---")
    rankings = extract_rankings(xl)
    rankings.to_csv(DATA_DIR / "rankings.csv", index=False)
    print(f"Saved to: {DATA_DIR / 'rankings.csv'}")

    # Save category reference
    print("\n--- Building Category Reference ---")
    categories = get_unique_categories(rankings)
    categories.to_csv(DATA_DIR / "categories.csv", index=False)
    print(f"Saved to: {DATA_DIR / 'categories.csv'}")

    # Summary
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE")
    print("="*50)
    print(f"Universities: {len(universities)}")
    print(f"Ranking records: {len(rankings)}")
    print(f"Unique categories: {len(categories)}")
    print(f"\nTop 10 categories by coverage:")
    print(categories.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
