"""
College Finder - Streamlit Web Application

A clean, user-friendly interface for exploring and ranking US colleges
based on subject preferences and other criteria.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from ranking_engine import RankingEngine

# Page config
st.set_page_config(
    page_title="US College Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_password():
    """Simple password protection for internal use."""

    # Check if password is configured in secrets
    try:
        has_password = "password" in st.secrets
    except Exception:
        # No secrets file exists - allow access (for local development)
        return True

    if not has_password:
        # Secrets file exists but no password configured
        return True

    # Check if already authenticated this session
    if st.session_state.get("authenticated", False):
        return True

    # Show login form
    st.markdown("## US College Finder")
    st.markdown("*Please log in to continue*")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Log in", type="primary"):
        if password == st.secrets["password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")

    return False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1.5rem;
        font-style: italic;
    }
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1E3A5F;
    }
    div[data-testid="stExpander"] details summary p {
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(ttl=3600)
def load_engine():
    """Load the ranking engine (cached for performance)."""
    data_dir = Path(__file__).parent / "data"
    return RankingEngine(
        universities_path=str(data_dir / "universities.csv"),
        rankings_path=str(data_dir / "rankings.csv")
    )


@st.cache_data
def load_categories():
    """Load category metadata."""
    data_dir = Path(__file__).parent / "data"
    return pd.read_csv(data_dir / "categories.csv")


def get_universities_from_engine(engine):
    """Get universities DataFrame from engine (avoids loading twice)."""
    return engine.universities


@st.cache_data(ttl=300)
def compute_rankings(_engine, selected_categories_tuple, selectivity_weight, coverage_penalty):
    """
    Cached ranking computation.
    Note: _engine is prefixed with _ to tell Streamlit not to hash it.
    selected_categories must be a tuple (hashable) for caching.
    """
    return _engine.compute_weighted_score(
        selected_categories=list(selected_categories_tuple),
        selectivity_factor=selectivity_weight,
        coverage_penalty=coverage_penalty
    )


def format_acceptance_rate(rate):
    if pd.isna(rate):
        return "N/A"
    return f"{rate*100:.1f}%"


def format_currency(val):
    if pd.isna(val):
        return "N/A"
    return f"${val:,.0f}"


def format_number(val):
    if pd.isna(val):
        return "N/A"
    return f"{val:,.0f}"


def main():
    engine = load_engine()
    categories_df = load_categories()
    universities_df = get_universities_from_engine(engine)  # Reuse from engine, don't load twice

    # Header
    st.markdown('<p class="main-header">US College Finder</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Discover the best-fit US universities based on your academic interests</p>', unsafe_allow_html=True)

    # Reorganized category groups
    category_groups = {
        "Overall Rankings": ["Global", "USA Overall", "Liberal Arts"],
        "Business & Economics": ["Business", "Economics", "Accounting & Finance"],
        "STEM - Computing & Engineering": ["Computer Science", "Engineering", "Mathematics"],
        "STEM - Sciences": ["Physics", "Chemistry", "Biology", "Natural Sciences", "Environmental Science"],
        "Engineering Specialties": ["Engineering (Mechanical)", "Engineering (Electrical)", "Engineering (Chemical)", "Engineering (Civil)"],
        "Health & Medicine": ["Pre-Med", "Medicine", "Nursing", "Public Health", "Pharmacy", "Dentistry"],
        "Humanities": ["English", "History", "Philosophy", "Classics", "Languages", "Religious Studies"],
        "Social Sciences": ["Psychology", "Political Science", "Sociology", "Anthropology", "Education", "Law", "Public Policy", "Geography"],
        "Arts & Media": ["Art & Design", "Architecture", "Film & Media", "Performing Arts"],
    }

    # Sidebar
    with st.sidebar:
        st.header("1. Select Academic Areas")

        # Clear all / Select all buttons
        col1, col2 = st.columns(2)
        with col1:
            clear_clicked = st.button("Clear All", use_container_width=True, key="clear_all_btn")
        with col2:
            select_all_clicked = st.button("Select All", use_container_width=True, key="select_all_btn")

        # Handle clear/select all
        if clear_clicked:
            for cat in engine.categories:
                st.session_state[f"sel_{cat}"] = False
            st.rerun()
        if select_all_clicked:
            for cat in engine.categories:
                st.session_state[f"sel_{cat}"] = True
            st.rerun()

        selected_categories = []

        for group_name, group_cats in category_groups.items():
            available_cats = [c for c in group_cats if c in engine.categories]
            if not available_cats:
                continue

            # Check current state of group
            group_selected_count = sum(1 for cat in available_cats if st.session_state.get(f"sel_{cat}", False))
            all_selected = group_selected_count == len(available_cats)

            with st.expander(f"{group_name} ({group_selected_count}/{len(available_cats)})",
                           expanded=(group_name in ["Overall Rankings", "STEM - Computing & Engineering", "Business & Economics"])):
                # Toggle all button for the group
                toggle_label = "Deselect all" if all_selected else "Select all"
                if st.button(toggle_label, key=f"grp_{group_name}", use_container_width=True):
                    new_state = not all_selected
                    for cat in available_cats:
                        st.session_state[f"sel_{cat}"] = new_state
                    st.rerun()

                # Individual checkboxes - use unique keys without default value conflict
                for cat in available_cats:
                    cat_info = categories_df[categories_df['category'] == cat]
                    count = cat_info['num_schools'].values[0] if len(cat_info) > 0 else 0

                    # Initialize state if not exists
                    if f"sel_{cat}" not in st.session_state:
                        st.session_state[f"sel_{cat}"] = False

                    if st.checkbox(f"{cat} ({count})", key=f"sel_{cat}"):
                        selected_categories.append(cat)

        st.divider()

        # Preferences
        st.header("2. Ranking Preferences")

        selectivity_weight = st.slider(
            "Selectivity Boost",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Boost more selective schools. 0 = pure subject ranking, 1 = heavily favor selective schools."
        )

        coverage_penalty = st.slider(
            "Breadth Requirement",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Require schools to have rankings across multiple selected areas. Higher = penalize specialist schools."
        )

        st.divider()

        # Filters
        st.header("3. Filters")

        acceptance_range = st.slider(
            "Acceptance Rate",
            min_value=0,
            max_value=100,
            value=(0, 100),
            format="%d%%"
        )

        all_states = sorted(universities_df['state'].dropna().unique())
        selected_states = st.multiselect(
            "States",
            options=all_states,
            default=[],
            placeholder="All states"
        )

    # Main content - Landing page when no categories selected
    if not selected_categories:
        st.info("👈 **Get started:** Select academic areas from the sidebar to see personalized college rankings.")

        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Universities", len(universities_df))
        with col2:
            st.metric("Academic Categories", len(categories_df))
        with col3:
            st.metric("Ranking Entries", f"{len(engine.rankings):,}")

        st.divider()

        # How it works section
        st.subheader("🎯 How It Works")
        how_cols = st.columns(4)
        with how_cols[0]:
            st.markdown("**1. Select Subjects**")
            st.caption("Choose academic areas you're interested in from the sidebar")
        with how_cols[1]:
            st.markdown("**2. Weighted Scoring**")
            st.caption("Each school gets a score (0-100) based on their subject rankings")
        with how_cols[2]:
            st.markdown("**3. Adjust Preferences**")
            st.caption("Optionally boost selective schools or require breadth")
        with how_cols[3]:
            st.markdown("**4. Explore Results**")
            st.caption("View rankings, compare schools, and export data")

        st.divider()

        # Two-column layout for About and Updates
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("📊 Data Sources")
            st.markdown("""
            **Rankings Data (2025-26):**
            - QS World University Rankings
            - Times Higher Education
            - US News & World Report
            - Niche College Rankings
            - CollegeVine Rankings

            **University Statistics:**
            - NCES IPEDS 2023-24
            - College Navigator
            - Common Data Sets
            """)

            st.subheader("💡 Tips")
            st.markdown("""
            - The **Zone** column shows selectivity (1 = most selective, 7 = most accessible)
            - Different zone scales exist for CS/Engineering, Business/Econ, and Pre-Med tracks
            - Use **Breadth Requirement** to find well-rounded schools ranked in multiple areas
            - Export your results to CSV for further analysis
            """)

        with right_col:
            st.subheader("📰 Recent Updates")
            st.markdown("""
            **February 2025 - Major Rankings Refresh**
            - Updated 50 subject categories to 2025-26 data
            - Added QS Subject Rankings 2025 (47 subjects)
            - Added QS Global Rankings 2026
            - Added Times Higher Ed Global & US National 2026
            - Added US News National & Liberal Arts 2026
            - Total: 6,100+ ranking entries across 56 categories

            **February 2025 - Performance**
            - Optimized memory usage for faster loading
            - Added caching for ranking computations

            **January 2025 - Launch**
            - 599 US universities
            - 56 academic subject categories
            - Custom weighted ranking algorithm
            """)

            st.subheader("🚧 Coming Soon")
            st.markdown("""
            - **Pre-Med, Public Health, Public Policy** rankings update (currently 2022 data)
            - **Niche rankings refresh** for 30+ subject categories
            - **University comparison** feature (side-by-side)
            - **Tuition & enrollment filters**
            - **Saved searches** and bookmarks
            """)

        st.divider()
        st.subheader("📚 Available Academic Categories")
        st.dataframe(
            categories_df[['category', 'num_schools', 'sources', 'latest_year']].rename(
                columns={'category': 'Subject Area', 'num_schools': 'Schools Ranked', 'sources': 'Data Sources', 'latest_year': 'Year'}
            ),
            hide_index=True,
            use_container_width=True
        )
        st.stop()

    # Compute rankings (cached based on parameters)
    with st.spinner("Computing rankings..."):
        results = compute_rankings(
            engine,
            tuple(sorted(selected_categories)),  # Tuple for hashability
            selectivity_weight,
            coverage_penalty
        )

    if results.empty:
        st.warning("No schools found matching your criteria.")
        st.stop()

    # Apply filters using boolean masks (avoids copy until needed)
    mask = pd.Series(True, index=results.index)

    if 'acceptance_rate' in results.columns:
        mask &= (
            results['acceptance_rate'].isna() |
            ((results['acceptance_rate'] >= acceptance_range[0]/100) &
             (results['acceptance_rate'] <= acceptance_range[1]/100))
        )

    if selected_states:
        mask &= results['state'].isin(selected_states)

    # Apply mask and re-rank (single copy here)
    filtered = results.loc[mask].reset_index(drop=True)
    filtered['display_rank'] = range(1, len(filtered) + 1)

    # Summary
    st.subheader(f"Results: {len(filtered)} Universities")
    st.caption(f"Ranked by: {', '.join(selected_categories)}")

    # Tabs layout (About info moved to landing page)
    tab1, tab2, tab3 = st.tabs(["Rankings", "University Details", "Export Data"])

    with tab1:
        # Clean table view - with Selectivity Zones prominently displayed
        display_cols = {
            'display_rank': 'Rank',
            'name': 'University',
            'selectivity_zone_general': 'Zone',
            'acceptance_rate': 'Accept %',
            'state': 'State',
            'sat_25_overall': 'SAT 25%',
            'sat_75_overall': 'SAT 75%',
            'undergrad_enrollment': 'Undergrads',
            'tuition_fees': 'Tuition',
        }

        available_display_cols = [c for c in display_cols.keys() if c in filtered.columns]
        display_df = filtered[available_display_cols].head(200).copy()
        display_df.columns = [display_cols[c] for c in available_display_cols]

        # Format columns
        if 'Accept %' in display_df.columns:
            display_df['Accept %'] = display_df['Accept %'].apply(format_acceptance_rate)
        if 'Zone' in display_df.columns:
            def format_zone(x):
                if pd.isna(x):
                    return ""
                if str(x) == 'True':
                    return "1"  # True = Zone 1 (most selective)
                try:
                    return f"{float(x):.1f}"
                except (ValueError, TypeError):
                    return str(x)
            display_df['Zone'] = display_df['Zone'].apply(format_zone)
        if 'Undergrads' in display_df.columns:
            display_df['Undergrads'] = display_df['Undergrads'].apply(format_number)
        if 'Tuition' in display_df.columns:
            display_df['Tuition'] = display_df['Tuition'].apply(format_currency)
        if 'SAT 25%' in display_df.columns:
            display_df['SAT 25%'] = display_df['SAT 25%'].apply(lambda x: int(x) if pd.notna(x) else "")
        if 'SAT 75%' in display_df.columns:
            display_df['SAT 75%'] = display_df['SAT 75%'].apply(lambda x: int(x) if pd.notna(x) else "")

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=550,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "University": st.column_config.TextColumn(width="large"),
                "Zone": st.column_config.TextColumn(width="small"),
            }
        )

        st.caption("**Zone** = Selectivity Zone (1=most selective, 7=most inclusive). See Export tab for subject-specific zones.")

    with tab3:
        st.subheader("Customize CSV Export")
        st.write("Select which columns to include in your export:")

        # Define column groups for export
        export_column_groups = {
            "Core Info (always included)": {
                'ipeds_id': 'IPEDS ID',
                'display_rank': 'Rank',
                'name': 'University',
            },
            "Selectivity Zones": {
                'selectivity_zone_general': 'Zone (General)',
                'selectivity_zone_cs_eng': 'Zone (CS/Engineering)',
                'selectivity_zone_econ_biz': 'Zone (Econ/Business)',
                'selectivity_zone_premed': 'Zone (Pre-Med)',
            },
            "Basic Info": {
                'website': 'Website',
                'description': 'Description',
                'popular_domains': 'Popular Majors',
                'uses_common_app': 'Uses Common App',
                'public_private': 'Public/Private',
            },
            "Location": {
                'city': 'City',
                'state': 'State',
                'region': 'Region',
                'campus_setting': 'Campus Setting',
            },
            "Admissions": {
                'acceptance_rate': 'Acceptance Rate',
                'total_applicants': 'Total Applicants',
                'yield_rate': 'Yield Rate',
                'ed_acceptance_rate': 'ED Acceptance Rate',
                'ed_to_rd_ratio': 'ED to RD Ratio',
            },
            "Test Scores": {
                'sat_25_overall': 'SAT 25th %ile',
                'sat_75_overall': 'SAT 75th %ile',
                'sat_math_25': 'SAT Math 25th',
                'sat_math_75': 'SAT Math 75th',
                'sat_english_25': 'SAT English 25th',
                'sat_english_75': 'SAT English 75th',
                'act_25': 'ACT 25th %ile',
                'act_75': 'ACT 75th %ile',
            },
            "Enrollment": {
                'undergrad_enrollment': 'Undergrad Enrollment',
                'grad_enrollment': 'Grad Enrollment',
                'pct_international': '% International',
                'pct_in_state': '% In-State',
            },
            "Cost & Outcomes": {
                'tuition_fees': 'Tuition & Fees',
                'room_board': 'Room & Board',
                'total_expenses': 'Total Expenses',
                'salary_10yr': 'Salary 10yr After',
            },
            "Sports (all columns)": {
                'sport_division': 'Sports Division',
                'total_athletes': 'Total Athletes',
                'pct_athletes': '% Athletes',
                'sports_revenue': 'Sports Revenue',
                'sports_expenses': 'Sports Expenses',
            },
        }

        # Add Subject Rankings as a special dynamic group (based on selected categories)
        include_subject_rankings = st.checkbox(
            "Subject Rankings (selected categories)",
            value=False,
            key="exp_subject_rankings",
            help="Include rank for each selected subject area"
        )

        # Row limit for export
        max_rows = st.number_input(
            "Max rows to export",
            min_value=10,
            max_value=len(filtered),
            value=min(150, len(filtered)),
            step=10,
            help="Limit the number of universities in the export"
        )

        # Default selections
        default_selected = [
            "Core Info (always included)",
            "Selectivity Zones",
            "Basic Info",
            "Location",
            "Admissions",
            "Test Scores",
            "Enrollment",
            "Cost & Outcomes",
        ]

        # Create checkboxes for each group
        selected_groups = []
        cols = st.columns(3)

        for i, (group_name, group_cols) in enumerate(export_column_groups.items()):
            with cols[i % 3]:
                is_core = group_name == "Core Info (always included)"
                default_val = group_name in default_selected
                if is_core:
                    st.checkbox(f"**{group_name}**", value=True, disabled=True, key=f"exp_{group_name}")
                    selected_groups.append(group_name)
                else:
                    if st.checkbox(group_name, value=default_val, key=f"exp_{group_name}"):
                        selected_groups.append(group_name)

        st.divider()

        # Build export dataframe based on selections
        export_cols_map = {}
        for group_name in selected_groups:
            export_cols_map.update(export_column_groups[group_name])

        # Filter to available columns and apply row limit
        available_export_cols = [c for c in export_cols_map.keys() if c in filtered.columns]
        export_df = filtered[available_export_cols].head(max_rows).copy()
        export_df.columns = [export_cols_map[c] for c in available_export_cols]

        # Add subject rankings if selected
        if include_subject_rankings:
            for cat in selected_categories:
                col_name = f"Rank: {cat}"
                export_df[col_name] = filtered.head(max_rows)['category_scores'].apply(
                    lambda x: x.get(cat, {}).get('rank', '') if isinstance(x, dict) else ''
                )

        # Format percentages - some are decimals (0.13), some are already percentages (13)
        # Columns stored as decimals (need *100)
        decimal_pct_cols = ['Acceptance Rate', 'Yield Rate', 'ED Acceptance Rate']
        for col in decimal_pct_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "")

        # Columns already stored as percentages (don't multiply)
        already_pct_cols = ['% International', '% In-State', '% Athletes']
        for col in already_pct_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "")

        # Format zones
        def format_zone_export(x):
            if pd.isna(x):
                return ""
            if str(x) == 'True':
                return "1"  # True = Zone 1 (most selective)
            try:
                return f"{float(x):.1f}"
            except (ValueError, TypeError):
                return str(x)

        zone_cols = ['Zone (General)', 'Zone (CS/Engineering)', 'Zone (Econ/Business)', 'Zone (Pre-Med)']
        for col in zone_cols:
            if col in export_df.columns:
                export_df[col] = export_df[col].apply(format_zone_export)

        # Preview
        st.write(f"**Preview** ({len(export_df)} rows, {len(export_df.columns)} columns):")
        st.dataframe(export_df.head(10), hide_index=True, use_container_width=True)

        # Download button
        csv = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="college_rankings.csv",
            mime="text/csv",
            type="primary"
        )

    with tab2:
        # University selector
        selected_uni = st.selectbox(
            "Select a university to view details",
            options=filtered['name'].tolist(),
            index=0
        )

        if selected_uni:
            uni_row = filtered[filtered['name'] == selected_uni].iloc[0]
            uni_full = universities_df[universities_df['ipeds_id'] == uni_row['ipeds_id']]

            if not uni_full.empty:
                uni_data = uni_full.iloc[0]

                # Four columns of info
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**Ranking Info**")
                    st.write(f"**Rank:** #{int(uni_row['display_rank'])} of {len(filtered)}")
                    st.write(f"**Ranked in:** {uni_row['num_categories']} of {len(selected_categories)} selected areas")

                with col2:
                    st.markdown("**Location**")
                    st.write(f"**City:** {uni_data.get('city', 'N/A')}")
                    st.write(f"**State:** {uni_data.get('state', 'N/A')}")
                    st.write(f"**Region:** {uni_data.get('region', 'N/A')}")
                    st.write(f"**Type:** {uni_data.get('public_private', 'N/A')}")
                    st.write(f"**Campus:** {uni_data.get('campus_setting', 'N/A')}")

                with col3:
                    st.markdown("**Admissions**")
                    st.write(f"**Acceptance Rate:** {format_acceptance_rate(uni_data.get('acceptance_rate'))}")
                    st.write(f"**ED Acceptance:** {format_acceptance_rate(uni_data.get('ed_acceptance_rate'))}")
                    st.write(f"**Applicants:** {format_number(uni_data.get('total_applicants'))}")
                    st.write(f"**Yield:** {format_acceptance_rate(uni_data.get('yield_rate'))}")
                    st.write(f"**SAT:** {format_number(uni_data.get('sat_25_overall'))} - {format_number(uni_data.get('sat_75_overall'))}")
                    st.write(f"**ACT:** {format_number(uni_data.get('act_25'))} - {format_number(uni_data.get('act_75'))}")

                with col4:
                    st.markdown("**Academics & Cost**")
                    st.write(f"**Undergrads:** {format_number(uni_data.get('undergrad_enrollment'))}")
                    st.write(f"**Tuition:** {format_currency(uni_data.get('tuition_fees'))}")
                    st.write(f"**Room & Board:** {format_currency(uni_data.get('room_board'))}")
                    st.write(f"**Total Cost:** {format_currency(uni_data.get('total_expenses'))}")
                    st.write(f"**10yr Salary:** {format_currency(uni_data.get('salary_10yr'))}")

                # Additional info row
                st.divider()
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("**Student Body**")
                    st.write(f"**% International:** {format_acceptance_rate(uni_data.get('pct_international'))}")
                    st.write(f"**% In-State:** {format_acceptance_rate(uni_data.get('pct_in_state'))}")

                with col2:
                    st.markdown("**Financial Aid**")
                    st.write(f"**% on Scholarship:** {format_acceptance_rate(uni_data.get('pct_scholarship'))}")
                    st.write(f"**Avg Scholarship:** {format_currency(uni_data.get('avg_scholarship'))}")

                with col3:
                    st.markdown("**Selectivity Zones**")
                    st.write(f"**General:** {uni_data.get('selectivity_zone_general', 'N/A')}")
                    st.write(f"**CS/Eng:** {uni_data.get('selectivity_zone_cs_eng', 'N/A')}")
                    st.write(f"**Econ/Biz:** {uni_data.get('selectivity_zone_econ_biz', 'N/A')}")
                    st.write(f"**Premed:** {uni_data.get('selectivity_zone_premed', 'N/A')}")

                with col4:
                    st.markdown("**Other**")
                    st.write(f"**Common App:** {uni_data.get('uses_common_app', 'N/A')}")
                    st.write(f"**Religious:** {uni_data.get('religious_affiliation', 'N/A')}")
                    if pd.notna(uni_data.get('website')):
                        st.write(f"[Visit Website](https://{uni_data['website']})")

                # Subject rankings for this school
                st.divider()
                st.markdown("**Subject Rankings (Selected Areas)**")
                cat_scores = uni_row.get('category_scores', {})
                if cat_scores:
                    rank_items = []
                    for cat, info in sorted(cat_scores.items()):
                        max_str = f" of {info['max_rank']}" if info.get('max_rank') else ""
                        rank_items.append(f"**{cat}:** #{info['rank']}{max_str}")

                    # Display in columns
                    cols = st.columns(4)
                    for i, item in enumerate(rank_items):
                        with cols[i % 4]:
                            st.write(item)
                else:
                    st.write("No subject ranking data available for selected categories.")

    # Footer
    st.divider()
    st.caption("Data sources: QS World University Rankings, US News, Times Higher Education, Niche, CollegeVine, NCES IPEDS | Rankings: 2025-26")


if __name__ == "__main__":
    if check_password():
        main()
