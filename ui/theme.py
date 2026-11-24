"""Orbital dark theme: page styling (CSS) and a matching Altair chart theme."""
import altair as alt
import streamlit as st

# Chart palette: categorical dark steps validated against the chart surface #0e1525
# (dataviz validator: all checks pass; worst adjacent CVD dE 8.4, normal-vision dE 19.3, all >= 3:1 contrast)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
GATE_COLOR = {"PASS": STATUS["good"], "REVIEW": STATUS["warning"], "HOLD": STATUS["critical"]}
INK, INK2, MUTED = "#e8eef8", "#b9c3d3", "#8792a6"
GRID, AXIS, SURFACE = "#1c2740", "#2b3a58", "#0e1525"
ACCENT = "#4fd1ff"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --ink:#e8eef8; --ink2:#b9c3d3; --muted:#8792a6; --accent:#4fd1ff; --surface:#0e1525;
        --line:rgba(120,170,255,0.14); --good:#0ca30c; --warn:#fab219; --crit:#d03b3b; }
html, body, .stApp, .stMarkdown, .stText, p, label, button, input, textarea, h1, h2, h3, h4, h5 { font-family: 'Space Grotesk', system-ui, sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded, span[class*="material"] { font-family: 'Material Symbols Rounded' !important; }
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(57,135,229,0.16), transparent 60%),
    radial-gradient(900px 500px at -10% 110%, rgba(144,133,233,0.12), transparent 60%),
    radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.35), transparent 100%),
    radial-gradient(1px 1px at 70% 80%, rgba(255,255,255,0.25), transparent 100%),
    radial-gradient(1px 1px at 40% 60%, rgba(255,255,255,0.2), transparent 100%),
    #05070d;
}
[data-testid="stHeader"] { background: rgba(5,7,13,0.72); backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"], [data-testid="stStatusWidget"],
footer, [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebar"] { background: rgba(9,13,24,0.92); border-right: 1px solid var(--line); }
h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--ink); }
.block-container { padding-top: 3.6rem; max-width: 1400px; }
/* glass cards (st.container(border=True)) */
[data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(180deg, rgba(20,30,52,0.55), rgba(12,18,32,0.55));
  border: 1px solid var(--line) !important; border-radius: 14px; }
[data-testid="stMetric"] { background: rgba(14,21,37,0.6); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.55rem; color: var(--ink); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricLabel"] p { color: var(--muted); text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.08em; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--accent) !important; }
.stButton > button { border: 1px solid rgba(79,209,255,0.45); background: rgba(79,209,255,0.08); color: var(--ink);
  border-radius: 10px; transition: all .15s ease; }
.stButton > button:hover { border-color: var(--accent); background: rgba(79,209,255,0.18); color: #fff; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
/* control panel toggle: always visible, accent styled */
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapseButton"] button {
  background: rgba(79,209,255,0.12) !important; border: 1px solid rgba(79,209,255,0.5) !important; border-radius: 8px !important; color: #4fd1ff !important; }
[data-testid="stSidebarCollapseButton"] { visibility: visible !important; opacity: 1 !important; }
.stButton > button[kind="tertiary"] { border: none; background: transparent; color: var(--accent); padding: 0; }
.stButton > button[kind="tertiary"]:hover { color: #fff; text-decoration: underline; background: transparent; }
.howto { font-size: .86rem; color: var(--ink2); background: rgba(14,21,37,.55); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 8px; padding: 8px 12px; margin: 4px 0 14px; line-height: 1.45; }
.stat { cursor: help; }
/* custom blocks */
.eyebrow { font-family:'JetBrains Mono',monospace; color: var(--accent); font-size: .72rem; letter-spacing: .22em; text-transform: uppercase; }
.hero-title { font-size: 2.5rem; font-weight: 700; line-height: 1.05; margin: .2rem 0 .3rem; color: var(--ink); }
.hero-sub { color: var(--ink2); font-size: 1.02rem; max-width: 900px; }
.kv { font-family:'JetBrains Mono',monospace; color: var(--ink); }
.muted { color: var(--muted); }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem; font-weight:600; border:1px solid; }
.pill.good { color:#3fd13f; border-color:rgba(12,163,12,.6); background:rgba(12,163,12,.12); }
.pill.warn { color:#fab219; border-color:rgba(250,178,25,.6); background:rgba(250,178,25,.10); }
.pill.crit { color:#ff6b6b; border-color:rgba(208,59,59,.7); background:rgba(208,59,59,.14); }
.flag { font-size:.82rem; color: var(--ink2); margin: 2px 0; }
.card-title { font-size: 1.15rem; font-weight: 600; color: var(--ink); margin: 0; }
.card-sub { font-size: .8rem; color: var(--muted); margin: 0 0 6px; }
.subbar { height: 6px; background: rgba(255,255,255,0.07); border-radius: 4px; overflow: hidden; }
.subbar > div { height: 100%; border-radius: 4px; }
.grid4 { display:grid; grid-template-columns: 1fr 1fr; gap: 6px 14px; margin: 6px 0; }
.lbl { font-size:.7rem; color: var(--muted); text-transform: uppercase; letter-spacing:.06em; display:flex; justify-content:space-between; }
.stat-row { display:flex; gap:18px; flex-wrap:wrap; margin: 6px 0 2px; }
.stat { min-width: 90px; } .stat .v { font-family:'JetBrains Mono',monospace; font-size:1.0rem; color: var(--ink); } .stat .k { font-size:.68rem; color: var(--muted); text-transform:uppercase; letter-spacing:.06em; }
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def _chart_theme():
    return {"config": {
        "background": "transparent",
        "view": {"stroke": "transparent"},
        "font": "system-ui, -apple-system, Segoe UI, sans-serif",
        "axis": {"labelColor": MUTED, "titleColor": INK2, "gridColor": GRID, "domainColor": AXIS,
                 "tickColor": AXIS, "labelFontSize": 11, "titleFontSize": 12, "titleFontWeight": 500},
        "legend": {"labelColor": INK2, "titleColor": INK2, "labelFontSize": 12},
        "title": {"color": INK, "fontSize": 14, "fontWeight": 600, "anchor": "start"},
        "range": {"category": SERIES},
    }}


try:
    alt.theme.register("orbital", enable=True)(_chart_theme)
except Exception:  # older Altair
    alt.themes.register("orbital", _chart_theme)
    alt.themes.enable("orbital")
