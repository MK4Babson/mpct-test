"""Light theme in Google's product style: page styling (CSS) and a matching Altair chart theme.

Text on white and #F8FAFD surfaces is #1F1F1F / #444746 / #5E6368 (all above 4.5:1). Google blue #0B57D0 marks actions.
Chart series use the validated light categorical palette (dataviz validator: worst adjacent CVD dE 9.1, normal-vision 19.6).
Status colours always come with an icon or a word, never colour alone.
"""
import altair as alt
import streamlit as st

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#188038", "warning": "#e37400", "serious": "#e8710a", "critical": "#d93025"}
GATE_COLOR = {"PASS": STATUS["good"], "REVIEW": STATUS["warning"], "HOLD": STATUS["critical"]}
INK, INK2, MUTED = "#1f1f1f", "#444746", "#5e6368"
GRID, AXIS, SURFACE = "#e8eaed", "#c4c7c5", "#ffffff"
ACCENT = "#0b57d0"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
:root { --ink:#1f1f1f; --ink2:#444746; --muted:#5e6368; --accent:#0b57d0; --surface:#ffffff; --bg:#f8fafd; --tint:#e8f0fe;
        --line:#e3e6ea; --good:#188038; --warn:#e37400; --crit:#d93025; }
html, body, .stApp, .stMarkdown, .stText, p, label, button, input, textarea, h1, h2, h3, h4, h5, li, td, th
  { font-family: 'Google Sans', 'Figtree', system-ui, -apple-system, 'Segoe UI', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #f8fafd; color: var(--ink); }
[data-testid="stHeader"] { background: rgba(255,255,255,0.96); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
footer, [data-testid="stDecoration"] { display: none !important; }
/* top navigation: always dark text on white, blue when active */
[data-testid="stTopNav"] *, [data-testid="stTopNavSection"] * { color: #1f1f1f !important; }
[data-testid="stTopNav"] a[aria-current="page"], [data-testid="stTopNav"] [aria-selected="true"] { color: #0b57d0 !important; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--ink); }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--ink); }
.block-container { padding-top: 3.6rem; max-width: 1400px; }
/* cards */
[data-testid="stVerticalBlockBorderWrapper"] { background: #ffffff; border: 1px solid var(--line) !important; border-radius: 18px;
  box-shadow: 0 1px 2px rgba(60,64,67,.08), 0 1px 3px 1px rgba(60,64,67,.06); }
[data-testid="stMetric"] { background: #ffffff; border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(60,64,67,.06); }
[data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 500; color: var(--ink); letter-spacing: -0.01em; }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricLabel"] p { color: var(--ink2); font-size: 0.84rem; font-weight: 500; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--ink2); font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border: 1px solid #747775; background: #ffffff; color: #0b57d0; border-radius: 999px; font-weight: 500; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #e8f0fe; border-color: #0b57d0; color: #0842a0; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"],
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {
  background: #0b57d0 !important; color: #ffffff !important; border-color: #0b57d0 !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 12px; background: #ffffff; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapseButton"] button {
  background: #e8f0fe !important; border: 1px solid #0b57d0 !important; border-radius: 10px !important; color: #0b57d0 !important; }
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; opacity: 1 !important; }
.stButton > button[kind="tertiary"] { border: none; background: transparent; color: var(--accent); padding: 0; box-shadow: none; }
.stButton > button[kind="tertiary"]:hover { color: #0842a0; text-decoration: underline; background: transparent; }
[data-testid="stExpander"] details { background: #ffffff; border: 1px solid var(--line); border-radius: 14px; }
.howto { font-size: .9rem; color: var(--ink2); background: #ffffff; border: 1px solid var(--line); border-left: 4px solid var(--accent);
  border-radius: 10px; padding: 10px 14px; margin: 4px 0 14px; line-height: 1.5; }
.stat { cursor: help; }
/* custom blocks */
.eyebrow { color: var(--accent); font-size: .78rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; }
.hero-title { font-size: 2.6rem; font-weight: 500; line-height: 1.1; margin: .25rem 0 .35rem; color: var(--ink); letter-spacing: -0.02em; }
.hero-sub { color: var(--ink2); font-size: 1.08rem; max-width: 900px; line-height: 1.5; }
.kv { font-family:'Roboto Mono',monospace; color: var(--ink); }
.muted { color: var(--muted); }
.pill { display:inline-block; padding:3px 11px; border-radius:999px; font-size:.78rem; font-weight:600; border:1px solid; }
.pill.good { color:#137333; border-color:#a8dab5; background:#e6f4ea; }
.pill.warn { color:#8a4b00; border-color:#fdd28a; background:#fef7e0; }
.pill.crit { color:#a50e0e; border-color:#f6aea9; background:#fce8e6; }
.flag { font-size:.86rem; color: var(--ink2); margin: 3px 0; }
.card-title { font-size: 1.2rem; font-weight: 600; color: var(--ink); margin: 0; }
.card-sub { font-size: .84rem; color: var(--muted); margin: 0 0 6px; }
.subbar { height: 7px; background: #e8eaed; border-radius: 4px; overflow: hidden; }
.subbar > div { height: 100%; border-radius: 4px; }
.grid4 { display:grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; margin: 6px 0; }
.lbl { font-size:.74rem; color: var(--ink2); font-weight: 500; text-transform: uppercase; letter-spacing:.05em; display:flex; justify-content:space-between; }
.stat-row { display:flex; gap:18px; flex-wrap:wrap; margin: 6px 0 2px; }
.stat { min-width: 90px; } .stat .v { font-size:1.08rem; font-weight: 600; color: var(--ink); } .stat .k { font-size:.72rem; color: var(--muted); text-transform:uppercase; letter-spacing:.05em; }
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def _chart_theme():
    return {"config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "font": "Figtree, system-ui, -apple-system, Segoe UI, sans-serif",
        "axis": {"labelColor": INK2, "titleColor": INK2, "gridColor": GRID, "domainColor": AXIS,
                 "tickColor": AXIS, "labelFontSize": 12, "titleFontSize": 13, "titleFontWeight": 500},
        "legend": {"labelColor": INK2, "titleColor": INK2, "labelFontSize": 12},
        "title": {"color": INK, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
        "range": {"category": SERIES},
    }}


try:
    alt.theme.register("google_light", enable=True)(_chart_theme)
except Exception:  # older Altair
    alt.themes.register("google_light", _chart_theme)
    alt.themes.enable("google_light")
