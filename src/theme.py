"""
Visual theme for the US College Finder: navy + gold on a warm cream background,
Poppins headings and Inter body text.

Pure presentation: inject_theme() adds a <style> block, page_header() renders
the banner. No app logic lives here.
"""

import html

import streamlit as st

NAVY = "#1B2A4A"
NAVY_DARK = "#111D35"
NAVY_LIGHT = "#2C4270"
GOLD = "#C4A24E"
GOLD_LIGHT = "#D4B76A"
GOLD_DARK = "#8B6914"
CREAM = "#F7F3ED"
CREAM_DARK = "#F0EAE0"
TEXT_BODY = "#3D4A5C"
TEXT_MUTED = "#6B7A8D"
BORDER = "#E2DDD4"
BORDER_LIGHT = "#EDE8E0"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@500;600;700&display=swap');

/* ---- Base ---- */
.stApp {{
    background-color: {CREAM};
    color: {TEXT_BODY};
}}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}
.block-container {{ padding-top: 2rem; }}
header[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3, h4, h5, h6 {{
    font-family: 'Poppins', system-ui, sans-serif !important;
    color: {NAVY} !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}}
.stMarkdown a {{ color: {GOLD_DARK}; text-decoration: none; font-weight: 500; }}
.stMarkdown a:hover {{ color: {GOLD}; text-decoration: underline; }}
.stMarkdown code {{
    background-color: {CREAM_DARK};
    color: {NAVY};
    border-radius: 3px;
}}
hr {{ border-color: {BORDER} !important; }}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background-color: #FFFFFF;
    border-right: 1px solid {BORDER_LIGHT};
}}
section[data-testid="stSidebar"] h2 {{
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_MUTED} !important;
    border-bottom: 2px solid {GOLD};
    padding-bottom: 0.4rem;
    margin-bottom: 0.5rem;
}}

/* ---- Buttons: primary solid navy, others outlined ---- */
.stButton > button, .stDownloadButton > button {{
    border-radius: 6px;
    font-weight: 600;
    transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {{
    box-shadow: 0 0 0 3px rgba(196, 162, 78, 0.35);
}}
button[data-testid="stBaseButton-primary"] {{
    background-color: {NAVY};
    border: 1px solid {NAVY};
    color: #FFFFFF;
}}
button[data-testid="stBaseButton-primary"]:hover {{
    background-color: {NAVY_LIGHT};
    border-color: {NAVY_LIGHT};
    color: #FFFFFF;
}}
button[data-testid="stBaseButton-secondary"] {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER};
    color: {NAVY};
}}
button[data-testid="stBaseButton-secondary"]:hover {{
    background-color: {CREAM};
    border-color: {GOLD};
    color: {NAVY};
}}

/* ---- Inputs: gold focus ---- */
.stTextInput [data-baseweb="input"]:focus-within,
.stTextArea [data-baseweb="textarea"]:focus-within,
.stNumberInput [data-baseweb="input"]:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stMultiSelect [data-baseweb="select"] > div:focus-within {{
    border-color: {GOLD} !important;
    box-shadow: 0 0 0 3px rgba(196, 162, 78, 0.15);
}}
.stMultiSelect [data-baseweb="tag"] {{
    background-color: {NAVY} !important;
    border-radius: 4px;
}}
.stMultiSelect [data-baseweb="tag"] span {{ color: #FFFFFF; }}

/* ---- Metrics as cards ---- */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    padding: 0.9rem 1.2rem;
    border-radius: 8px;
    border: 1px solid {BORDER_LIGHT};
    border-left: 4px solid {GOLD};
    box-shadow: 0 1px 3px rgba(27, 42, 74, 0.06);
}}
[data-testid="stMetricLabel"] p {{
    font-size: 0.72rem !important;
    color: {TEXT_MUTED} !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Poppins', sans-serif;
    color: {NAVY} !important;
    font-weight: 600 !important;
}}

/* ---- Dataframes and expanders as cards ---- */
[data-testid="stDataFrame"] {{
    border-radius: 8px;
    border: 1px solid {BORDER_LIGHT};
    box-shadow: 0 1px 3px rgba(27, 42, 74, 0.06);
    overflow: hidden;
}}
[data-testid="stExpander"] details {{
    background: #FFFFFF;
    border: 1px solid {BORDER_LIGHT} !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(27, 42, 74, 0.05);
}}
[data-testid="stExpander"] details summary p {{
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    color: {NAVY};
}}
[data-testid="stExpander"] details summary:hover p {{ color: {GOLD_DARK}; }}

/* ---- Tabs: gold underline on active ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.5rem;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] p {{
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    color: {TEXT_MUTED};
}}
.stTabs [aria-selected="true"] p {{ color: {NAVY}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {GOLD} !important; }}

/* ---- Alerts: softer, on-palette ---- */
[data-testid="stAlert"] > div {{ border-radius: 8px; }}

/* ---- Page banner ---- */
.cf-banner {{
    background: linear-gradient(135deg, #1E3050 0%, {NAVY_DARK} 60%, #0D1520 100%);
    border-radius: 12px;
    padding: 2.1rem 2.25rem 1.9rem;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(27, 42, 74, 0.12);
}}
.cf-banner::before {{
    content: '';
    position: absolute;
    top: -40%; right: -8%;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(212, 183, 106, 0.14) 0%, transparent 70%);
    pointer-events: none;
}}
.cf-banner::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {GOLD}, {GOLD_LIGHT}, transparent);
}}
.cf-banner .cf-eyebrow {{
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {GOLD_LIGHT};
    margin: 0 0 0.5rem 0;
}}
.cf-banner h1 {{
    color: {GOLD_LIGHT} !important;
    font-size: 2.3rem !important;
    font-weight: 600 !important;
    line-height: 1.15 !important;
    margin: 0 0 0.5rem 0 !important;
    padding: 0 !important;
}}
.cf-banner .cf-sub {{
    color: #B8C1D0;
    font-size: 1rem;
    max-width: 680px;
    margin: 0;
    line-height: 1.55;
}}
.cf-banner .cf-stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.25rem;
}}
.cf-banner .cf-stat {{
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(212, 183, 106, 0.28);
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    color: #D8DEE8;
}}
.cf-banner .cf-stat b {{ color: #FFFFFF; font-weight: 600; }}
@media (max-width: 640px) {{
    .cf-banner {{ padding: 1.5rem 1.25rem; }}
    .cf-banner h1 {{ font-size: 1.7rem !important; }}
}}
</style>
"""


def inject_theme() -> None:
    """Inject the theme CSS. Call once per run, right after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "",
                stats: list[tuple[str, str]] | None = None) -> None:
    """Render the navy banner.

    `stats` is a list of (value, label) pairs shown as pills under the subtitle.
    """
    eyebrow_html = f'<p class="cf-eyebrow">{html.escape(eyebrow)}</p>' if eyebrow else ""
    sub_html = f'<p class="cf-sub">{html.escape(subtitle)}</p>' if subtitle else ""
    stats_html = ""
    if stats:
        pills = "".join(
            f'<span class="cf-stat"><b>{html.escape(v)}</b> {html.escape(k)}</span>'
            for v, k in stats
        )
        stats_html = f'<div class="cf-stats">{pills}</div>'
    st.markdown(
        f'<div class="cf-banner">{eyebrow_html}<h1>{html.escape(title)}</h1>{sub_html}{stats_html}</div>',
        unsafe_allow_html=True,
    )
