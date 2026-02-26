# US College Finder

A Streamlit web application for ranking and comparing US universities based on academic subject preferences.

## Features

- **Subject-based ranking**: Select academic areas to find best-fit universities
- **Selectivity zones**: 1-7 scale rating system for admissions difficulty
- **Customizable exports**: Choose columns for CSV download
- **Filters**: By acceptance rate, state, and more

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deployed on Streamlit Community Cloud. To add password protection, add to Streamlit secrets:

```toml
password = "your-password"
```

## Data

- `data/universities.csv` - 599 US universities with admissions, enrollment, and outcome data
- `data/rankings.csv` - Subject rankings from QS, US News, THE, Niche, CollegeVine
- `data/categories.csv` - 56 academic categories

## Data Sources

- QS World University Rankings
- US News & World Report
- Times Higher Education
- Niche
- CollegeVine
- NCES IPEDS
- BigJ Educational Consultancy
