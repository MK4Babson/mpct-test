"""
Multi-Prime Commercial Control Tower: web app.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> point it at this repo, main file app.py

Pages:  Mission Control (globe + programs) -> Insights (project health cards) -> Project (deep dive)
        Program analytics (all manager views and evidence)
"""
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path as _Path

# Keep the light theme even when .streamlit/config.toml is missing (web uploads to GitHub often skip folders whose
# names start with a dot). The file is recreated on first run; Streamlit applies it after one reload.
_cfg = _Path(__file__).resolve().parent / ".streamlit" / "config.toml"
if not _cfg.exists():
    try:
        _cfg.parent.mkdir(exist_ok=True)
        _cfg.write_text('[theme]\nbase = "light"\nprimaryColor = "#0b57d0"\nbackgroundColor = "#f8fafd"\n'
                        'secondaryBackgroundColor = "#ffffff"\ntextColor = "#1f1f1f"\nfont = "sans serif"\n\n'
                        '[client]\ntoolbarMode = "minimal"\n')
    except OSError:
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from config import TOLERANCES, PER_DIEM_CAP  # noqa: E402
from run_engine import run, scorecard  # noqa: E402
from schema import SCHEMA, coerce, read_folder  # noqa: E402
from validate_input import validate  # noqa: E402
from insights import (filter_scope, project_summary, prime_trend, rate_spread,  # noqa: E402
                      generate_insights, prime_names, money, RULE_LABEL)
import manager as M  # noqa: E402
from health import health_table  # noqa: E402

st.set_page_config(page_title="Multi-Prime Commercial Control Tower", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="collapsed")
from ui import theme  # noqa: E402  (registers the chart theme)
from ui import pages as P  # noqa: E402
from ui import analytics_pages as A  # noqa: E402
from ui import cost_pages as CP  # noqa: E402
from ui import scm_pages as SCM  # noqa: E402
from ui import macro_pages as GM  # noqa: E402
from ui import limits_page as LIM  # noqa: E402
from ui import payapp_pages as PAY  # noqa: E402
from ui import front_page as FP  # noqa: E402
import decisions as DEC  # noqa: E402
from ui import pattern_pages as PP  # noqa: E402
from ui import ops_pages as OP  # noqa: E402
from ui import schedule_page as SP  # noqa: E402
from ui.theme import SERIES  # noqa: E402

theme.apply()

DATASETS = {
    "Demo: West Texas data center program (synthetic)": ROOT / "data" / "raw",
    "Demo: North Koel canal program (India, built from real tenders)": ROOT / "data" / "north_koel",
    "Demo: clean invoices (use with stress test)": ROOT / "data" / "clean_demo",
}
PROGRAM_META = {
    "Demo: West Texas data center program (synthetic)": ("West Texas data center program", "West Texas, USA · fictional campuses", "datacenter"),
    "Demo: North Koel canal program (India, built from real tenders)": ("North Koel canal program", "Aurangabad district, Bihar, India", "canal"),
    "Demo: clean invoices (use with stress test)": ("Clean demo program", "West Texas, USA · fictional campuses", "datacenter"),
}
HOME_PROGRAMS = list(DATASETS)[:2]
GATE_LABEL = {"PASS": "✓ PASS", "REVIEW": "! REVIEW", "HOLD": "✕ HOLD"}
LEVEL = {"critical": ":red[**✕ Act now**]", "warning": ":orange[**! Watch**]",
         "info": ":blue[**i Plan**]", "good": ":green[**✓ Good news**]"}
STATUS = {"Red": "🔴 Act now", "Amber": "🟠 Watch", "Green": "🟢 On track"}
DOLLAR = st.column_config.NumberColumn(format="$%,.0f")


def templates_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for t, cols in SCHEMA.items():
            z.writestr(f"{t}.csv", ",".join(c for c, *_ in cols) + "\n")
        for t, cols in M.CONTROL_SCHEMA.items():
            z.writestr(f"{t}.csv", ",".join(c for c, *_ in cols) + "\n")
        dd = ROOT / "DATA_DICTIONARY.md"
        if dd.exists():
            z.writestr("DATA_DICTIONARY.md", dd.read_text())
    return buf.getvalue()


def go(page, source=None, project=None):
    """Button callback: switch program and/or project, then jump to a page."""
    if source:
        st.session_state["program"] = source
        st.session_state["source_radio"] = source
    if project:
        st.session_state["project"] = project
    st.session_state["_goto"] = page


# ---------------- loading + computing ----------------
def load_folder(folder):
    t = read_folder(folder)
    gt = folder / "ground_truth_errors.csv"
    if gt.exists():
        t["ground_truth_errors"] = pd.read_csv(gt, dtype={"invoice_id": str})
    return t, M.read_controls(folder)


EXTRA_NAMES = ["co_details", "approval_log"]


def load_extras_folder(folder):
    return {n: pd.read_csv(folder / f"{n}.csv") for n in EXTRA_NAMES if (folder / f"{n}.csv").exists()}


def load_uploads(uploaded):
    tables, ctl = {}, {}
    for f in uploaded:
        name = Path(f.name).stem
        if name in EXTRA_NAMES:
            st.session_state.setdefault("upload_extras", {})[name] = pd.read_csv(f)
            continue
        if name in SCHEMA or name == "ground_truth_errors":
            str_cols = {c: str for c, typ, *_ in SCHEMA.get(name, []) if typ == "str"}
            tables[name] = pd.read_csv(f, dtype=str_cols or {"invoice_id": str})
        elif name in M.CONTROL_SCHEMA:
            ctl[name] = pd.read_csv(f, dtype={"project_id": str, "invoice_id": str, "period": str})
        else:
            st.sidebar.warning(f"Ignored {f.name}: file name doesn't match a table name.")
    return tables, ctl


def prepare(raw, raw_ctl):
    tables = {k: coerce(v, k) for k, v in raw.items() if k in SCHEMA}
    if "projects" in raw:
        tables["projects"] = raw["projects"]  # keep extra columns: short_name, coordinates, structure
    return tables, M.coerce_controls(raw_ctl or {})


@st.cache_data(show_spinner="Checking invoices...")
def run_cached(_tables, key, tol_items, cap):
    return run(tolerances=dict(tol_items), tables=_tables, per_diem_cap=cap)


def table_key(name, tables):
    return (name, tuple(sorted((k, len(v), float(pd.util.hash_pandas_object(v, index=False).sum())) for k, v in tables.items())))


def compute(tables, controls, exceptions, gates, f_projects, f_primes, f_periods):
    """Everything the pages need for one program and one filter selection."""
    names = prime_names(tables)
    labels = M.project_labels(tables, controls)
    end_period = f_periods[-1]
    gf, ef, ids = filter_scope(tables, gates, exceptions, f_projects, f_primes, f_periods)
    summary = project_summary(tables, gates, ids, f_projects, f_primes, f_periods)
    trend = prime_trend(gf)
    spread = rate_spread(tables, f_projects, f_primes)
    pt, prog = M.program_table(tables, gates, exceptions, controls, f_projects, f_primes, end_period, labels)
    risk = M.split_risk(gf, ef)
    q = gf.merge(risk, on="invoice_id", how="left").groupby("project_id").agg(
        first_pass_pct=("gate", lambda s: round(100 * (s == "PASS").mean(), 0)),
        bills_on_hold=("gate", lambda s: int((s == "HOLD").sum())),
        blocked=("blocked", "sum"), to_check=("to_check", "sum"))
    pt = pt.merge(q, left_on="project_id", right_index=True, how="left")
    rc = M.root_causes(gf, ef)
    perf = M.score_contractors(M.contractor_performance(gf, ef, pt), names, pt)
    bench_items, bench_idx = M.rate_benchmark(tables, f_projects, f_primes, labels)
    cash, overdue = M.cash_view(tables, gates, exceptions, controls, f_projects, f_primes, end_period, pt)

    def relabel(d):
        for k in ("title", "detail", "action"):
            for pid in sorted(labels, key=len, reverse=True):
                d[k] = d[k].replace(pid, labels[pid])
        return d

    ins = [relabel(d) for d in generate_insights(tables, gf, ef, summary, trend, spread, top_n=6)]
    ins += M.manager_insights(pt, rc, perf, bench_idx, cash, overdue)
    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    ins.sort(key=lambda d: (order[d["level"]], -d["amount"]))
    health = health_table(pt, cash)
    return dict(names=names, labels=labels, end_period=end_period, gf=gf, ef=ef, summary=summary, trend=trend,
                spread=spread, pt=pt, prog=prog, risk=risk, rc=rc, perf=perf, bench_items=bench_items,
                bench_idx=bench_idx, cash=cash, overdue=overdue, insights=ins[:12], health=health,
                has_ctl="project_controls" in controls, has_pay="payments" in controls)


@st.cache_data(show_spinner="Scanning programs from orbit...")
def snapshot(source):
    """Default-settings summary of a demo program for Mission Control."""
    raw_, ctl_ = load_folder(DATASETS[source])
    tables_, controls_ = prepare(raw_, ctl_)
    ex_, g_ = run(tables=tables_)
    inv = tables_["invoices"]
    c = compute(tables_, controls_, ex_, g_, sorted(inv.project_id.unique()), sorted(inv.prime_id.unique()),
                sorted(inv.period.unique()))
    pr = tables_["projects"].set_index("project_id")
    ps = c["pt"].merge(c["health"], on="project_id")
    ps["latitude"] = ps.project_id.map(pr.latitude)
    ps["longitude"] = ps.project_id.map(pr.longitude)
    ps["location"] = ps.project_id.map(pr.location)
    w = ps.committed
    hl = (ps.health.fillna(0) * w).sum() / w.sum()
    flags = [f for fl in ps.sort_values("health")["alerts"] for f in fl[:1]]
    label, region, structure = PROGRAM_META[source]
    from health import status_of
    return {"source": source, "label": label, "region": region, "structure": structure,
            "projects": ps[["project_id", "name", "latitude", "longitude", "location", "health", "health_status"]],
            "committed": float(ps.committed.sum()), "progress": float(ps.verified.sum() / ps.committed.sum()),
            "blocked": float(ps.blocked.fillna(0).sum()), "health": float(hl), "status": status_of(hl), "flags": flags}


# ---------------- Sidebar: data ----------------
# the chosen program lives in a plain session key so it survives page switches
SOURCES = list(DATASETS) + ["Upload my own CSV files"]
if st.session_state.get("program") not in SOURCES:
    import os
    _d = int(os.environ.get("MPCT_DEFAULT_PROGRAM", "0"))  # testing aid: start on another demo program
    st.session_state["program"] = SOURCES[_d if 0 <= _d < len(SOURCES) else 0]


def _pick_source():
    st.session_state["program"] = st.session_state["source_radio"]


ROLES = {
    "Program manager (everything)": None,
    "Executive": ["front", "home", "insights", "schedule", "markets", "story", "trends", "mobile", "limits"],
    "Cost controller": ["front", "home", "insights", "project", "issues", "payapp", "cost", "co", "forecast", "schedule", "materials", "markets", "process", "trends", "analytics", "limits"],
    "Accounts payable": ["front", "home", "insights", "issues", "payapp", "analytics", "intake", "process", "mobile", "limits"],
    "Contractor manager": ["front", "home", "insights", "issues", "contractors", "patterns", "co", "schedule", "materials", "ml", "intake", "limits"],
    "Materials & procurement": ["front", "home", "insights", "materials", "markets", "schedule", "cost", "limits"],
    "Sustainability lead": ["front", "home", "insights", "carbon", "story", "limits"],
}
ROLE_FOCUS = {
    "Executive": "Health, the five-minute story and trends.",
    "Cost controller": "Budget, forecasts, fair prices, change orders and cash.",
    "Accounts payable": "Bills on hold, intake, cycle time and the decision queue.",
    "Contractor manager": "Contractor quality, patterns, change orders and the review queue.",
    "Sustainability lead": "Embodied carbon and the low-carbon what-ifs.",
    "Materials & procurement": "Stock everywhere, the material plan, lookahead constraints, suppliers, tariffs and FX.",
}
with st.sidebar:
    st.markdown('<div class="eyebrow">Control panel</div>', unsafe_allow_html=True)
    role = st.selectbox("Role view (demo)", list(ROLES), key="role", help="Shows the pages each role would use. This is a view filter for the demo, "
                        "not access control: a real deployment needs sign-in and server-side permissions.")
    if role in ROLE_FOCUS:
        st.caption(ROLE_FOCUS[role])
    st.header("1. Data")
    st.radio("Choose data", SOURCES, index=SOURCES.index(st.session_state["program"]), key="source_radio", on_change=_pick_source)
    source = st.session_state["program"]
    uploaded = None
    if source.startswith("Upload"):
        uploaded = st.file_uploader("Upload the 7 required CSVs. Optional: primes, projects, project_controls, "
                                    "baseline_plan, payments (they unlock budget, schedule and cash views).",
                                    type="csv", accept_multiple_files=True)
        st.download_button("Download blank templates", templates_zip(), "mpct_templates.zip")

if source in DATASETS:
    raw, raw_ctl = load_folder(DATASETS[source])
    extras = load_extras_folder(DATASETS[source])
elif uploaded:
    raw, raw_ctl = load_uploads(uploaded)
    extras = dict(st.session_state.get("upload_extras", {}))
else:
    st.info("Upload your CSV files in the sidebar. Start from the blank templates; DATA_DICTIONARY.md explains every column.")
    st.stop()

issues = validate(raw)
errors = [i for i in issues if i["level"] == "ERROR"]
if errors:
    with st.expander(f"Input check: {len(errors)} error(s), {len(issues) - len(errors)} warning(s)", expanded=True):
        st.dataframe(pd.DataFrame(issues), width="stretch", hide_index=True)
    st.error("Fix the input errors above, then upload again.")
    st.stop()

tables, controls = prepare(raw, raw_ctl)
names = prime_names(tables)
labels = M.project_labels(tables, controls)
plab = lambda p: labels.get(p, p)  # noqa: E731

# ---------------- Sidebar: filters + tolerances ----------------
inv_all = tables["invoices"]
all_projects = sorted(inv_all.project_id.dropna().unique())
all_primes = sorted(inv_all.prime_id.dropna().unique())
all_periods = sorted(inv_all.period.dropna().unique())
sk = abs(hash(source)) % 10**6
with st.sidebar:
    st.header("2. Filters")
    f_projects = st.multiselect("Projects", all_projects, default=all_projects, format_func=plab, key=f"fp{sk}")
    f_primes = st.multiselect("Contractors", all_primes, default=all_primes, format_func=lambda p: names.get(p, p), key=f"fc{sk}")
    if len(all_periods) > 1:
        p0, p1 = st.select_slider("Months", options=all_periods, value=(all_periods[0], all_periods[-1]), key=f"fm{sk}")
        f_periods = [p for p in all_periods if p0 <= p <= p1]
    else:
        f_periods = all_periods
    st.caption("Billing-quality figures use the months selected. Budget, schedule and cash figures are "
               "cumulative to the last month selected.")
    st.header("3. Tolerances")
    with st.expander("How strict each check is", expanded=False):
        tol = dict(TOLERANCES)
        tol["qty_review_pct"] = st.slider("Qty over verified: REVIEW above (%)", 0.0, 10.0, TOLERANCES["qty_review_pct"], 0.25)
        tol["qty_hold_pct"] = st.slider("Qty over verified: HOLD above (%)", 0.0, 20.0, TOLERANCES["qty_hold_pct"], 0.5)
        st.caption("Rate over contract has no tolerance: any rate above the contract is held. Only a rounding "
                   "allowance of half a cent per unit applies, because a contract rate is a legal number.")
        tol["pct_review_pts"] = st.slider("Lump-sum % over verified: REVIEW above (pts)", 0.0, 10.0, TOLERANCES["pct_review_pts"], 0.5)
        tol["pct_hold_pts"] = st.slider("Lump-sum % over verified: HOLD above (pts)", 0.0, 20.0, TOLERANCES["pct_hold_pts"], 0.5)
        tol["tm_hold_pct"] = st.slider("T&M hours over approved: HOLD above (%)", 0.0, 30.0, TOLERANCES["tm_hold_pct"], 1.0)
        cap = st.number_input("Per diem cap ($/worker-day)", value=PER_DIEM_CAP, step=1.0)
        st.caption("Try the Policy simulator in Program analytics to compare whole policies side by side.")

if not (f_projects and f_primes and f_periods):
    st.warning("Pick at least one project, one contractor and one month.")
    st.stop()

data_key = table_key(source, tables)
exceptions, gates = run_cached(tables, data_key, tuple(sorted(tol.items())), cap)
C = compute(tables, controls, exceptions, gates, f_projects, f_primes, f_periods)
end_period = C["end_period"]
gf, ef, summary, trend, spread = C["gf"], C["ef"], C["summary"], C["trend"], C["spread"]
pt, prog, risk, rc, perf = C["pt"], C["prog"], C["risk"], C["rc"], C["perf"]
bench_items, bench_idx, cash, overdue = C["bench_items"], C["bench_idx"], C["cash"], C["overdue"]
insights, has_ctl, has_pay = C["insights"], C["has_ctl"], C["has_pay"]

# context for the new pages
pr_tab = tables.get("projects", pd.DataFrame(columns=["project_id"])).set_index("project_id")
_get = lambda col, p, default=None: (pr_tab[col].get(p, default) if col in pr_tab.columns else default)  # noqa: E731
meta = PROGRAM_META.get(source, ("Your program", "Uploaded data", "datacenter"))
snaps = [snapshot(s) for s in HOME_PROGRAMS]


@st.cache_data(show_spinner="Re-running each month-end for the trend view...")
def monthly(_tables, _controls, _ex, _g, key, projects, primes, periods):
    rows = []
    for i, p in enumerate(periods):
        c = compute(_tables, _controls, _ex, _g, list(projects), list(primes), list(periods[: i + 1]))
        pt_, h = c["pt"], c["health"]
        w = pt_.set_index("project_id").committed
        hl = (h.set_index("project_id").health.reindex(w.index).fillna(0) * w).sum() / w.sum() if w.sum() else np.nan
        comm = pt_.committed.sum()
        ver = 100 * pt_.verified.sum() / comm if comm else np.nan
        plan = (pt_.planned_pct * pt_.committed).sum() / comm if comm and pt_.planned_pct.notna().any() else np.nan
        rows.append({"period": p, "health": hl, "first_pass": 100 * (c["gf"].gate == "PASS").mean() if len(c["gf"]) else np.nan,
                     "blocked": c["risk"].blocked.sum(), "to_check": c["risk"].to_check.sum(), "ver_minus_plan": ver - plan if pd.notna(plan) else np.nan,
                     "vac": pt_.vac.sum() if "vac" in pt_ and pt_.vac.notna().any() else np.nan,
                     "at_risk_sites": int((h.health_status == "At risk").sum())})
    return pd.DataFrame(rows)


ctx = SimpleNamespace(
    **C, tables=tables, gates=gates, exceptions=exceptions, go=go, snapshots=snaps, program_label=meta[0], region=meta[1],
    controls=controls, extras=extras, data_folder=DATASETS.get(source), source_key="".join(ch for ch in meta[0] if ch.isalnum())[:30] or "program",
    monthly=lambda: monthly(tables, controls, exceptions, gates, (data_key, tuple(sorted(tol.items())), cap), tuple(f_projects), tuple(f_primes), tuple(f_periods)),
    structure_of=lambda p: _get("structure", p, meta[2]) if isinstance(_get("structure", p), str) else meta[2],
    location_of=lambda p: _get("location", p, meta[1]) if isinstance(_get("location", p), str) else meta[1],
    full_name_of=lambda p: _get("project_name", p, labels.get(p, p)) if isinstance(_get("project_name", p), str) else labels.get(p, p),
    coords_of=lambda p: (float(_get("latitude", p, np.nan)), float(_get("longitude", p, np.nan))),
)


def analytics():
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        st.button("← Insights", key="an_back", on_click=go, args=("insights", None, None), type="tertiary")
    with b2:
        st.button("⌂ Mission Control", key="an_home", on_click=go, args=("home", None, None), type="tertiary")
    st.markdown(f'<div class="eyebrow">Program analytics · {meta[0]}</div><div class="hero-title">Commercial control</div>'
                '<div class="hero-sub">Cost, schedule, cash and billing control: every view, every number, and the evidence behind it.</div>',
                unsafe_allow_html=True)
    if issues:
        with st.expander(f"Input check: {len(issues)} warning(s)"):
            st.dataframe(pd.DataFrame(issues), width="stretch", hide_index=True)

    # ---------------- KPIs ----------------
    st.markdown("##### Cost and schedule")
    k = st.columns(6)
    budget = pt.budget.sum() if has_ctl else np.nan
    eac, vac = pt.eac.sum(), (pt.vac.sum() if has_ctl else np.nan)
    committed = pt.committed.sum()
    k[0].metric("Approved budget", money(budget) if has_ctl else "-",
                help="Owner's approved budget incl. contingency (project_controls.csv)")
    k[1].metric("Forecast at completion", money(eac),
                delta=(f"{money(abs(vac))} {'under' if vac >= 0 else 'over'} budget" if has_ctl else None),
                delta_color="normal" if (has_ctl and vac >= 0) else "inverse",
                help="Committed contract value (with approved change orders) plus pending change orders")
    k[2].metric("Billed to date", money(pt.billed.sum()), f"{100 * pt.billed.sum() / committed:.0f}% of committed" if committed else None,
                delta_color="off", delta_arrow="off")
    ver = 100 * pt.verified.sum() / committed if committed else np.nan
    plan = (pt.planned_pct * pt.committed).sum() / committed if committed and pt.planned_pct.notna().any() else np.nan
    k[3].metric("Verified complete", f"{ver:.0f}%", f"{ver - plan:+.0f} pts vs plan ({plan:.0f}%)" if pd.notna(plan) else None,
                delta_color="normal")
    late = int((pt.delay_weeks > 0).sum()) if has_ctl else 0
    k[4].metric("Projects late", f"{late} of {len(pt)}" if has_ctl else "-")
    k[5].metric("Delay penalty exposure", money(pt.ld_exposure.sum()) if has_ctl else "-",
                help="What the owner could recover as liquidated damages if forecast delays happen (rate x weeks, capped)")

    st.markdown("##### Billing control")
    k = st.columns(6)
    k[0].metric("✓ First-pass rate", f"{100 * (gf.gate == 'PASS').mean():.0f}%" if len(gf) else "-",
                help="Share of bills with no issue at all")
    k[1].metric("✕ Flagged: recommend hold", money(risk.blocked.sum()), f"{(gf.gate == 'HOLD').sum()} of {len(gf)} bills",
                delta_color="off", delta_arrow="off", help="Amount flagged by checks that recommend holding payment. "
                "A recommendation, not a decision: an authorized reviewer decides, under the contract.")
    k[2].metric("! Flagged: needs a look", money(risk.to_check.sum()), f"{(gf.gate == 'REVIEW').sum()} bill{'s' if (gf.gate == 'REVIEW').sum() != 1 else ''}",
                delta_color="off", delta_arrow="off", help="Amount on checks that need a look but don't recommend a hold")
    k[3].metric("Pending change orders", money(pt.pending_co.sum()))
    k[4].metric("Paid, needs recovery", money(cash.paid_needs_recovery.sum()) if has_pay and "paid_needs_recovery" in cash else "-",
                help="Flagged amounts that were already paid; recover them from the next bill")
    if has_pay and "avg_days_to_pay" in cash and cash.avg_days_to_pay.notna().any():
        k[5].metric("Avg days to pay", f"{cash.avg_days_to_pay.mean():.0f}", f"terms {cash.terms.mean():.0f} days", delta_color="off", delta_arrow="off")
    else:
        k[5].metric("Avg days to pay", "-")
    flagged = float(risk.blocked.sum() + risk.to_check.sum())
    dm = DEC.measures(ctx.source_key)
    if dm["decided"]:
        ladder = (f'<b>3. Confirmed adjustment</b> {money(dm["confirmed"])} and <b>4. Money saved</b> {money(dm["saved"])}: '
                  f'from {dm["decided"]} recorded decisions on the Issues & decisions page (all checks, not just this view).')
    else:
        ladder = ('<b>3. Confirmed adjustment</b> and <b>4. Money saved</b>: none yet. They come only from recorded reviewer '
                  'decisions on the Issues & decisions page.')
    st.markdown(
        '<div class="howto"><b>From flag to saving.</b> These are four different numbers and are never added together. '
        f'<b>1. Flagged</b> {money(flagged)}: what the bill checks found in this view. '
        '<b>2. Under review</b>: flagged amounts waiting for evidence or a decision. '
        + ladder + ' A flag is not a saving.</div>', unsafe_allow_html=True)
    if not has_ctl:
        st.info("Add project_controls.csv, baseline_plan.csv and payments.csv to unlock budget, schedule, penalty and cash figures "
                "(templates in the sidebar).")

    tabs = st.tabs(["Manager overview", "This week's decisions", "Root causes", "Contractors", "Schedule & penalties",
                    "Cash flow", "Policy simulator", "Invoices", "Exceptions", "Accuracy & testing"])

    # ---------------- Manager overview ----------------
    with tabs[0]:
        st.subheader("Program health")
        st.caption("One row per project. Status combines cost, schedule and billing gaps; 'Why' says what drives it.")
        show = pt.assign(status_=pt.status.map(STATUS))
        cols = ["name", "contractor", "status_", "why", "budget", "eac", "vac", "planned_pct", "verified_pct", "spi",
                "forecast_finish", "delay_weeks", "ld_exposure", "first_pass_pct", "blocked", "to_check"]
        st.dataframe(show[cols], width="stretch", hide_index=True, column_config={
            "name": "Project", "contractor": "Contractor", "status_": "Status", "why": "Why",
            "budget": st.column_config.NumberColumn("Budget", format="$%,.0f"),
            "eac": st.column_config.NumberColumn("Forecast cost", format="$%,.0f"),
            "vac": st.column_config.NumberColumn("Under (+) / over (-) budget", format="$%,.0f"),
            "planned_pct": st.column_config.NumberColumn("Planned %", format="%.0f"),
            "verified_pct": st.column_config.NumberColumn("Verified %", format="%.0f"),
            "spi": st.column_config.NumberColumn("Schedule index", format="%.2f", help="Verified % / planned %. Below 1 = behind"),
            "forecast_finish": st.column_config.DateColumn("Forecast finish"),
            "delay_weeks": st.column_config.NumberColumn("Weeks late", format="%.1f"),
            "ld_exposure": st.column_config.NumberColumn("Penalty exposure", format="$%,.0f"),
            "first_pass_pct": st.column_config.NumberColumn("First-pass %", format="%.0f"),
            "blocked": st.column_config.NumberColumn("Blocked", format="$%,.0f"),
            "to_check": st.column_config.NumberColumn("To check", format="$%,.0f")})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Budget vs forecast vs billed** ($ millions)")
            meas = (["Budget"] if has_ctl else []) + ["Forecast cost", "Billed", "Verified work"]
            long = pt.rename(columns={"budget": "Budget", "eac": "Forecast cost", "billed": "Billed", "verified": "Verified work"})
            long = long.melt(id_vars="name", value_vars=meas, var_name="measure", value_name="v")
            long["v"] = long.v / 1e6
            ch = (alt.Chart(long).mark_bar(cornerRadiusEnd=4, stroke="#ffffff", strokeWidth=2)
                  .encode(x=alt.X("name:N", title=None, axis=alt.Axis(labelAngle=0)),
                          xOffset=alt.XOffset("measure:N", sort=meas),
                          y=alt.Y("v:Q", title="$ millions"),
                          color=alt.Color("measure:N", sort=meas, scale=alt.Scale(domain=meas, range=SERIES[:len(meas)]),
                                          legend=alt.Legend(orient="top", title=None)),
                          tooltip=["name", "measure", alt.Tooltip("v:Q", format=",.2f", title="$M")])
                  .properties(height=320))
            st.altair_chart(ch, width="stretch")
        with c2:
            st.markdown("**Progress over time: planned vs verified vs billed** (% of committed value)")
            pick = st.selectbox("Show", ["Whole program"] + list(pt.name), label_visibility="collapsed")
            pr = prog[prog.project_id.isin(pt.project_id)].copy()
            if pick != "Whole program":
                pr = pr[pr.project_id == pt.set_index("name").project_id[pick]]
            agg = pr.groupby("period").apply(lambda d: pd.Series({
                "Planned": (d.planned_pct * d.committed).sum() / d.committed.sum() if d.planned_pct.notna().any() else np.nan,
                "Verified": 100 * d.verified.sum() / d.committed.sum() if d.verified.notna().any() else np.nan,
                "Billed": 100 * d.billed.sum() / d.committed.sum() if d.billed.notna().any() else np.nan}),
                include_groups=False).reset_index()
            sl = agg.melt(id_vars="period", var_name="series", value_name="pct").dropna()
            dom = ["Planned", "Verified", "Billed"]
            line = (alt.Chart(sl).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2)
                    .encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=-45)),
                            y=alt.Y("pct:Q", title="% complete", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[SERIES[3], SERIES[2], SERIES[0]]),
                                            legend=alt.Legend(orient="top", title=None)),
                            strokeDash=alt.condition(alt.datum.series == "Planned", alt.value([5, 4]), alt.value([1, 0])),
                            tooltip=["series", "period", alt.Tooltip("pct:Q", format=".1f")])
                    .properties(height=290))
            rule = alt.Chart(pd.DataFrame({"period": [end_period]})).mark_rule(color="#888", strokeDash=[2, 2]).encode(x="period:O")
            st.altair_chart(line + rule, width="stretch")
            st.caption("Gap between Billed and Verified = paid ahead of work in place. Gap below Planned = behind schedule. "
                       "Dotted line = last month selected.")

    # ---------------- Decisions ----------------
    with tabs[1]:
        st.subheader("This week's decisions")
        st.caption("Cost, schedule, cash and billing findings for the current selection, ranked by urgency and dollar impact. "
                   "Every hold here is a recommendation: the contract, the evidence and an authorized reviewer decide whether "
                   "payment is withheld, with written notice to the contractor.")
        if not insights:
            st.success("Nothing needs a decision in this view.")
        brief = ["# This week's decisions", "",
                 f"Scope: {len(f_projects)} project(s), {len(f_primes)} contractor(s), {f_periods[0]} to {f_periods[-1]}", ""]
        for i, d in enumerate(insights, 1):
            with st.container(border=True):
                c1, c2 = st.columns([1, 7])
                c1.markdown(LEVEL[d["level"]])
                c2.markdown(f"**{d['title']}**  \n{d['detail']}  \n*Recommended action:* {d['action']}")
            brief += [f"{i}. **{d['title']}**", f"   - {d['detail']}", f"   - Action: {d['action']}", ""]
        st.download_button("Download this brief (Markdown)", "\n".join(brief), "decisions_brief.md")

    # ---------------- Root causes ----------------
    with tabs[2]:
        st.subheader("Why bills fail, and what fixing each cause is worth")
        if not len(rc):
            st.success("No failed checks in this view.")
        else:
            st.caption(f"First-pass rate now: **{rc.attrs['first_pass_now']:.0f}%**. Causes are ranked by how many bills they "
                       "touch. 'First-pass if fixed' is cumulative: fix this cause and every cause above it.")
            c1, c2 = st.columns(2)
            rc_sorted = rc.root_cause.tolist()
            b1 = (alt.Chart(rc).mark_bar(cornerRadiusEnd=4, color=SERIES[1], height=18)
                  .encode(y=alt.Y("root_cause:N", sort=rc_sorted, title=None, axis=alt.Axis(labelLimit=260)),
                          x=alt.X("invoices_affected:Q", title="Bills affected"),
                          tooltip=["root_cause", "owner", "invoices_affected", alt.Tooltip("money_at_risk:Q", format="$,.0f")])
                  .properties(height=alt.Step(30), title="Bills affected by each cause"))
            c1.altair_chart(b1, width="stretch")
            b2 = (alt.Chart(rc).mark_bar(cornerRadiusEnd=4, color=SERIES[2], height=18)
                  .encode(y=alt.Y("root_cause:N", sort=rc_sorted, title=None, axis=alt.Axis(labels=False, ticks=False)),
                          x=alt.X("first_pass_if_fixed:Q", title="First-pass rate if fixed (%)", scale=alt.Scale(domain=[0, 100])),
                          tooltip=["root_cause", "first_pass_if_fixed"])
                  .properties(height=alt.Step(30), title="First-pass rate after fixing down to this cause"))
            c2.altair_chart(b2, width="stretch")
            st.dataframe(rc[["root_cause", "check_failed", "owner", "process_fix", "invoices_affected", "pct_of_failed_invoices",
                             "money_at_risk", "first_pass_if_fixed"]], width="stretch", hide_index=True, column_config={
                "root_cause": "Root cause", "check_failed": "Check that failed", "owner": "Who owns the fix",
                "process_fix": "Process fix", "invoices_affected": "Bills", "pct_of_failed_invoices": "% of failed bills",
                "money_at_risk": st.column_config.NumberColumn("Money at risk", format="$%,.0f"),
                "first_pass_if_fixed": st.column_config.NumberColumn("First-pass if fixed %", format="%.0f")})

    # ---------------- Contractors ----------------
    with tabs[3]:
        st.subheader("Contractor performance")
        if len(perf):
            st.caption("Score = 40% first-pass rate + 30% billing accuracy (money at risk vs billed) + 30% schedule index. "
                       "Below 60, or worsening and below 80 = performance meeting.")
            st.dataframe(perf[["contractor", "score", "action", "first_pass_pct", "trend", "risk_pct_of_billed", "schedule_spi",
                               "invoices", "billed", "blocked", "to_check", "repeat_issues"]],
                         width="stretch", hide_index=True, column_config={
                             "contractor": "Contractor", "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
                             "action": "Action", "first_pass_pct": "First-pass %", "trend": "Trend",
                             "risk_pct_of_billed": "% of billing at risk", "schedule_spi": "Schedule index", "invoices": "Bills",
                             "billed": st.column_config.NumberColumn("Billed", format="$%,.0f"),
                             "blocked": st.column_config.NumberColumn("Blocked", format="$%,.0f"),
                             "to_check": st.column_config.NumberColumn("To check", format="$%,.0f"),
                             "repeat_issues": "Issues repeated in 2+ months"})
        if len(trend) and trend.period.nunique() > 1:
            tr = trend.assign(contractor=trend.prime_id.map(lambda p: names.get(p, p)))
            dom = [names.get(p, p) for p in all_primes]
            line = (alt.Chart(tr).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2)
                    .encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=0)),
                            y=alt.Y("first_pass_pct:Q", title="First-pass rate (%)", scale=alt.Scale(domain=[0, 100])),
                            color=alt.Color("contractor:N", scale=alt.Scale(domain=dom, range=SERIES[:len(dom)]),
                                            legend=alt.Legend(orient="top", title=None)),
                            tooltip=["contractor", "period", "first_pass_pct"])
                    .properties(height=280, title="First-pass rate by month (higher is better)"))
            st.altair_chart(line, width="stretch")

        st.subheader("Rate benchmarking")
        if len(bench_idx):
            st.caption("Each project's contract rates compared with the program median for the same items. "
                       "100 = median; 106 = paying 6% more for the same work.")
            c1, c2 = st.columns([2, 3])
            bi = (alt.Chart(bench_idx).mark_bar(cornerRadiusEnd=4, height=18)
                  .encode(y=alt.Y("project:N", sort="-x", title=None),
                          x=alt.X("price_index:Q", title="Price index (median = 100)", scale=alt.Scale(zero=False, domain=[
                              min(90, bench_idx.price_index.min() - 2), max(110, bench_idx.price_index.max() + 2)])),
                          color=alt.condition(alt.datum.price_index > 100, alt.value(SERIES[1]), alt.value(SERIES[0])),
                          tooltip=["project", alt.Tooltip("price_index:Q", format=".1f"),
                                   alt.Tooltip("excess_vs_median:Q", format="$,.0f"), "items_compared"])
                  .properties(height=alt.Step(32)))
            rl = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(color="#888", strokeDash=[3, 3]).encode(x="x:Q")
            c1.altair_chart(bi + rl, width="stretch")
            c2.dataframe(bench_items.head(12)[["item_code", "description", "unit", "projects", "lowest", "median", "highest",
                                               "highest_at", "spread_pct", "excess_vs_median"]], width="stretch", hide_index=True,
                         column_config={"lowest": DOLLAR, "median": DOLLAR, "highest": DOLLAR,
                                        "excess_vs_median": st.column_config.NumberColumn("Paid above median", format="$%,.0f"),
                                        "spread_pct": "Spread %", "highest_at": "Highest at"})
        else:
            st.caption("Needs the same item code on at least two projects.")

        st.markdown("**Where each contractor's money at risk comes from**")
        e = ef[ef.amount_at_risk > 0].merge(gf[["invoice_id", "prime_id"]], on="invoice_id")
        if len(e):
            piv = (e.assign(issue=e.rule.map(lambda r: RULE_LABEL.get(r, r)), contractor=e.prime_id.map(lambda p: names.get(p, p)))
                   .pivot_table(index="issue", columns="contractor", values="amount_at_risk", aggfunc="sum", fill_value=0))
            st.dataframe(piv.round(0), width="stretch", column_config={c: DOLLAR for c in piv.columns})
        with st.expander("Billing scorecard (raw counts)"):
            st.dataframe(scorecard(gf, ef, tables.get("primes")), width="stretch", hide_index=True,
                         column_config={"billed_gross": DOLLAR, "amount_at_risk": DOLLAR})

    # ---------------- Schedule & penalties ----------------
    with tabs[4]:
        st.subheader("Schedule and delay penalties")
        if not has_ctl:
            st.info("Add project_controls.csv (planned dates, penalty terms) and baseline_plan.csv to see this view.")
        else:
            st.caption("Delay penalty (liquidated damages) = weekly rate x weeks late, capped, applied to the committed contract "
                       "value. It is money the owner can recover, but a late project usually costs the owner far more than "
                       "the penalty pays back.")
            g = pt.dropna(subset=["planned_finish"]).copy()
            ps = controls["project_controls"].set_index("project_id").planned_start
            g["planned_start"] = g.project_id.map(ps)
            bars = []
            for r in g.itertuples():
                bars.append({"name": r.name, "part": "Planned", "start": r.planned_start, "end": r.planned_finish})
                if pd.notna(r.forecast_finish) and r.forecast_finish > r.planned_finish:
                    bars.append({"name": r.name, "part": "Forecast delay", "start": r.planned_finish, "end": r.forecast_finish})
            gb = pd.DataFrame(bars)
            dd = g.data_date.dropna().max()
            gantt = (alt.Chart(gb).mark_bar(cornerRadius=3, height=16)
                     .encode(y=alt.Y("name:N", title=None, sort=list(g.sort_values("planned_finish").name)),
                             x=alt.X("start:T", title=None), x2="end:T",
                             color=alt.Color("part:N", scale=alt.Scale(domain=["Planned", "Forecast delay"], range=[SERIES[0], SERIES[7]]),
                                             legend=alt.Legend(orient="top", title=None)),
                             tooltip=["name", "part", alt.Tooltip("start:T", format="%d %b %Y"), alt.Tooltip("end:T", format="%d %b %Y")])
                     .properties(height=alt.Step(34)))
            layers = gantt
            if pd.notna(dd):
                layers = gantt + alt.Chart(pd.DataFrame({"d": [dd]})).mark_rule(color="#555", strokeDash=[3, 3]).encode(x="d:T")
            st.altair_chart(layers, width="stretch")
            st.caption("Dashed line = data date.")
            st.dataframe(g[["name", "contractor", "planned_pct", "verified_pct", "spi", "planned_finish", "forecast_finish",
                            "forecast_source", "delay_weeks", "ld_rate", "ld_cap", "committed", "ld_exposure"]],
                         width="stretch", hide_index=True, column_config={
                             "name": "Project", "contractor": "Contractor",
                             "planned_pct": st.column_config.NumberColumn("Planned %", format="%.0f"),
                             "verified_pct": st.column_config.NumberColumn("Verified %", format="%.0f"),
                             "spi": st.column_config.NumberColumn("Schedule index", format="%.2f"),
                             "planned_finish": st.column_config.DateColumn("Contract finish"),
                             "forecast_finish": st.column_config.DateColumn("Forecast finish"),
                             "forecast_source": "Forecast from", "delay_weeks": st.column_config.NumberColumn("Weeks late", format="%.1f"),
                             "ld_rate": st.column_config.NumberColumn("Penalty %/week", format="%.2f"),
                             "ld_cap": st.column_config.NumberColumn("Cap %", format="%.0f"),
                             "committed": st.column_config.NumberColumn("Contract value", format="$%,.0f"),
                             "ld_exposure": st.column_config.NumberColumn("Penalty exposure", format="$%,.0f")})
            st.caption("'Earned schedule' = the planned finish moved by how far behind (or ahead of) the plan the work is today, in time. "
                       "See Forecast & risk for the backtest behind this choice and the range of likely finish dates.")

    # ---------------- Cash flow ----------------
    with tabs[5]:
        st.subheader("Cash flow")
        if not len(cash):
            st.info("No bills in this view.")
        else:
            c = st.columns(5)
            c[0].metric("Paid to date", money(cash.paid_to_date.sum()) if "paid_to_date" in cash else "-")
            c[1].metric("Billed, not yet paid", money(cash.unpaid_amount.sum()) if "unpaid_amount" in cash else "-")
            c[2].metric("Overdue (past terms)", money(cash.overdue_amount.sum()) if "overdue_amount" in cash else "-")
            c[3].metric("Retention held", money(cash.retention_held.sum()),
                        help="Withheld from every bill until completion; released at the end (owner liability)")
            c[4].metric("Next month's cash need", money(cash.next_month_net.sum()),
                        help="Average of the last 3 bills per project, capped by what's left on the contract, less retention "
                             "and advance recovery")
            if not has_pay:
                st.info("Add payments.csv to see paid amounts, payment days and overdue bills.")
            show_cols = [c for c in ["name", "billed_gross", "paid_to_date", "unpaid_amount", "overdue_amount", "avg_days_to_pay",
                                     "terms", "paid_within_terms_pct", "retention_held", "paid_needs_recovery",
                                     "next_month_net"] if c in cash]
            st.dataframe(cash[show_cols], width="stretch", hide_index=True, column_config={
                "name": "Project", "billed_gross": st.column_config.NumberColumn("Billed", format="$%,.0f"),
                "paid_to_date": st.column_config.NumberColumn("Paid", format="$%,.0f"),
                "unpaid_amount": st.column_config.NumberColumn("Not yet paid", format="$%,.0f"),
                "overdue_amount": st.column_config.NumberColumn("Overdue", format="$%,.0f"),
                "avg_days_to_pay": st.column_config.NumberColumn("Avg days to pay", format="%.0f"),
                "terms": st.column_config.NumberColumn("Terms (days)", format="%.0f"),
                "paid_within_terms_pct": st.column_config.NumberColumn("% paid on time", format="%.0f"),
                "retention_held": st.column_config.NumberColumn("Retention held", format="$%,.0f"),
                "paid_needs_recovery": st.column_config.NumberColumn("Paid, needs recovery", format="$%,.0f"),
                "next_month_net": st.column_config.NumberColumn("Next month (net)", format="$%,.0f")})
            if "advance_paid" in cash and cash.advance_paid.fillna(0).sum() > 0:
                st.markdown("**Mobilization advance recovery**")
                st.caption("The advance is recovered from bills as work progresses, between the agreed start and end points "
                           "(here 10% to 80% of contract value). A shortfall means the owner is carrying the contractor's money.")
                a = cash.dropna(subset=["advance_paid"])
                al = a.melt(id_vars="name", value_vars=["advance_paid", "advance_expected_recovered", "advance_recovered"],
                            var_name="m", value_name="v")
                mlab = {"advance_paid": "Advance paid", "advance_expected_recovered": "Should be recovered by now",
                        "advance_recovered": "Actually recovered"}
                al["m"] = al.m.map(mlab)
                dom = list(mlab.values())
                ab = (alt.Chart(al).mark_bar(cornerRadiusEnd=4, stroke="#ffffff", strokeWidth=2)
                      .encode(x=alt.X("name:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset=alt.XOffset("m:N", sort=dom),
                              y=alt.Y("v:Q", title="$"), color=alt.Color("m:N", sort=dom, scale=alt.Scale(domain=dom, range=SERIES[:3]),
                                                                        legend=alt.Legend(orient="top", title=None)),
                              tooltip=["name", "m", alt.Tooltip("v:Q", format="$,.0f")])
                      .properties(height=280))
                st.altair_chart(ab, width="stretch")
                st.dataframe(a[["name", "progress_pct_of_contract", "advance_paid", "advance_expected_recovered",
                                "advance_recovered", "advance_shortfall"]], width="stretch", hide_index=True, column_config={
                    "name": "Project", "progress_pct_of_contract": st.column_config.NumberColumn("Progress %", format="%.0f"),
                    "advance_paid": st.column_config.NumberColumn("Advance paid", format="$%,.0f"),
                    "advance_expected_recovered": st.column_config.NumberColumn("Should be recovered", format="$%,.0f"),
                    "advance_recovered": st.column_config.NumberColumn("Recovered", format="$%,.0f"),
                    "advance_shortfall": st.column_config.NumberColumn("Shortfall", format="$%,.0f")})
            if len(overdue):
                st.markdown("**Bills unpaid beyond terms**")
                st.dataframe(overdue.assign(project=overdue.project_id.map(plab)).drop(columns="project_id"),
                             width="stretch", hide_index=True, column_config={"net_amount": st.column_config.NumberColumn("Net amount", format="$%,.0f")})

    # ---------------- Policy simulator ----------------
    with tabs[6]:
        st.subheader("Policy simulator")
        st.markdown("Compare whole checking policies on the current selection. **Stricter** catches more overbilling but "
                    "holds more bills (more reviewer work, slower payments). **Looser** pays faster but lets money slip through.")

        def scope_fn(g_, ex_):
            g2, e2, _ = filter_scope(tables, g_, ex_, f_projects, f_primes, f_periods)
            return g2, e2

        truth = raw.get("ground_truth_errors")
        sim = M.policy_scenarios(lambda t_: run_cached(tables, data_key, tuple(sorted(t_.items())), cap), tol, scope_fn, truth)
        st.dataframe(sim, width="stretch", hide_index=True, column_config={
            "policy": "Policy", "bills_on_hold": "Bills on hold", "bills_to_review": "Bills to review",
            "first_pass_pct": st.column_config.NumberColumn("First-pass %", format="%.0f"),
            "money_blocked": st.column_config.NumberColumn("Money blocked", format="$%,.0f"),
            "money_to_check": st.column_config.NumberColumn("Money to check", format="$%,.0f"),
            "errors_caught": "Planted errors caught", "false_alarms": "False alarms",
            "overbilling_missed": st.column_config.NumberColumn("Overbilling missed", format="$%,.0f")})
        sl = sim.melt(id_vars="policy", value_vars=["bills_on_hold", "bills_to_review"], var_name="k", value_name="n")
        sl["k"] = sl.k.map({"bills_on_hold": "On hold", "bills_to_review": "To review"})
        pol_order = list(sim.policy)
        c1, c2 = st.columns(2)
        wl = (alt.Chart(sl).mark_bar(stroke="#ffffff", strokeWidth=2)
              .encode(y=alt.Y("policy:N", sort=pol_order, title=None, axis=alt.Axis(labelLimit=260)), x=alt.X("n:Q", title="Bills needing a person"),
                      color=alt.Color("k:N", scale=alt.Scale(domain=["On hold", "To review"], range=[SERIES[7], SERIES[3]]),
                                      legend=alt.Legend(orient="top", title=None)),
                      tooltip=["policy", "k", "n"])
              .properties(height=alt.Step(36), title="Workload"))
        c1.altair_chart(wl, width="stretch")
        if "overbilling_missed" in sim:
            mb = (alt.Chart(sim).mark_bar(cornerRadiusEnd=4, color=SERIES[1], height=20)
                  .encode(y=alt.Y("policy:N", sort=pol_order, title=None, axis=alt.Axis(labels=False, ticks=False)),
                          x=alt.X("overbilling_missed:Q", title="Overbilling missed ($)"),
                          tooltip=["policy", alt.Tooltip("overbilling_missed:Q", format="$,.0f"), "errors_caught"])
                  .properties(height=alt.Step(36), title="Money that slips through"))
            c2.altair_chart(mb, width="stretch")
        else:
            c2.info("Missed overbilling needs an answer key (ground_truth_errors.csv), or run the stress test.")

    # ---------------- Invoices ----------------
    with tabs[7]:
        st.caption("Evidence: every bill with its gate. Open one to see its lines and the checks that failed.")
        f = st.multiselect("Show gates", ["PASS", "REVIEW", "HOLD"], default=["HOLD", "REVIEW"])
        view = gf.merge(risk, on="invoice_id", how="left")
        view = view[view.gate.isin(f)].sort_values("amount_at_risk", ascending=False)
        view.insert(3, "project", view.project_id.map(plab))
        st.dataframe(view, width="stretch", hide_index=True)
        if len(view):
            pick = st.selectbox("Open an invoice", view.invoice_id)
            st.markdown(f"**{pick}**: {GATE_LABEL[view.set_index('invoice_id').at[pick, 'gate']]}")
            lines = tables["invoice_lines"]
            st.dataframe(lines[lines.invoice_id == pick], width="stretch", hide_index=True)
            st.dataframe(ef[ef.invoice_id == pick], width="stretch", hide_index=True)

    # ---------------- Exceptions ----------------
    with tabs[8]:
        rules = sorted(ef.rule.unique()) if len(ef) else []
        pick_rules = st.multiselect("Rules", rules, default=rules)
        ev = ef[ef.rule.isin(pick_rules)]
        st.dataframe(ev, width="stretch", hide_index=True)
        st.download_button("Download exceptions (CSV)", ev.to_csv(index=False), "exceptions.csv")
        st.download_button("Download invoice gates (CSV)", gf.to_csv(index=False), "invoice_gates.csv")

    # ---------------- Accuracy & testing ----------------
    with tabs[9]:
        st.markdown("#### Accuracy against the answer key")
        if "ground_truth_errors" in raw:
            truth = raw["ground_truth_errors"]
            if len(truth) == 0:
                st.info("This dataset has an empty answer key (no planted errors). Every exception here would be a false alarm.")
                st.metric("False alarms", len(exceptions))
            else:
                def ks(df, col):
                    return set(zip(df.invoice_id.astype(str), pd.to_numeric(df.line_no, errors="coerce").fillna(-1).astype(int), df[col]))
                det, tru = ks(exceptions, "rule"), ks(truth, "error_type")
                acc = []
                for r in sorted({x[2] for x in tru}):
                    d, t_ = {x for x in det if x[2] == r}, {x for x in tru if x[2] == r}
                    tp = len(d & t_)
                    acc.append({"rule": r, "planted": len(t_), "caught": tp, "missed": len(t_ - d),
                                "false_alarms": len(d - t_), "recall_%": round(100 * tp / len(t_), 1),
                                "precision_%": round(100 * tp / len(d), 1) if d else None})
                bad = set(truth.invoice_id.astype(str))
                gb = gates.assign(bad=gates.invoice_id.astype(str).isin(bad))
                n_bad, n_h, n_r, n_p = int(gb.bad.sum()), int((gb.bad & (gb.gate == "HOLD")).sum()), \
                    int((gb.bad & (gb.gate == "REVIEW")).sum()), int((gb.bad & (gb.gate == "PASS")).sum())
                n_fa = int((~gb.bad & (gb.gate != "PASS")).sum())
                b = st.columns(4)
                b[0].metric("Bills with a planted error", n_bad)
                b[1].metric("Sent to recommend-hold", n_h)
                b[2].metric("Sent to review", n_r)
                b[3].metric("Missed (passed)", n_p, f"{n_fa} clean bills flagged", delta_color="off", delta_arrow="off")
                st.caption(f"{n_h} + {n_r} = {n_h + n_r} of {n_bad} bad bills were stopped for a person to look at. "
                           "Caution: the checks and the planted errors were built together, so a perfect score here proves the code "
                           "runs, not that it works on real bills. The stress test below and the tolerance sweep are the harder tests.")
                st.dataframe(pd.DataFrame(acc), width="stretch", hide_index=True)
                st.caption("Covers the whole dataset, not just the filtered view. Move the tolerance sliders to see the trade-off.")
        else:
            st.info("No answer key (ground_truth_errors.csv) in this dataset. Use the stress test below to create one.")

        st.markdown("#### Stress test")
        st.markdown("Plants known errors into a **copy** of the current data, reruns the engine, and reports "
                    "what was caught, what was missed, and any new false alarms. Use it to test real data that has no answer key.")
        c1, c2, c3 = st.columns(3)
        n_err = c1.number_input("Errors to plant", 5, 200, 40, 5)
        subtle = c2.slider("Share of tiny errors", 0.0, 1.0, 0.25, 0.05)
        seed = c3.number_input("Random seed", 1, 9999, 1)
        if st.button("Run stress test"):
            with tempfile.TemporaryDirectory() as tmp:
                src, dst = Path(tmp) / "data", Path(tmp) / "stress"
                src.mkdir()
                for name, df in tables.items():
                    if name in SCHEMA:
                        coerce(df, name).to_csv(src / f"{name}.csv", index=False)
                res = subprocess.run([sys.executable, str(ROOT / "src" / "inject_errors.py"), "--data", str(src),
                                      "--out", str(dst), "--n", str(n_err), "--subtle", str(subtle), "--seed", str(seed)],
                                     capture_output=True, text=True, cwd=str(ROOT / "src"))
                if res.returncode != 0:
                    st.error(res.stderr[-2000:])
                else:
                    report = (dst / "stress_test_report.md").read_text()
                    st.markdown(report.replace("\n## ", "\n##### ").replace("# Stress test report", "#### Stress test report"))



PAGES = {
    "front": st.Page(lambda: FP.page(ctx), title="This month's decisions", icon=":material/task_alt:", url_path="decisions", default=True),
    "home": st.Page(lambda: P.home(ctx), title="Mission Control", icon=":material/public:", url_path="mission"),
    "insights": st.Page(lambda: P.insights(ctx), title="Insights", icon=":material/grid_view:", url_path="insights"),
    "project": st.Page(lambda: P.project(ctx), title="Project", icon=":material/apartment:", url_path="project"),
    "forecast": st.Page(lambda: A.forecast(ctx), title="Forecast & risk", icon=":material/insights:", url_path="forecast"),
    "ml": st.Page(lambda: A.ml_lab(ctx), title="ML Lab", icon=":material/model_training:", url_path="ml-lab"),
    "schedule": st.Page(lambda: SP.page(ctx), title="Schedule (CPM/PERT)", icon=":material/account_tree:", url_path="schedule"),
    "cost": st.Page(lambda: CP.cost_intel(ctx), title="Cost intelligence", icon=":material/price_check:", url_path="cost"),
    "co": st.Page(lambda: CP.change_orders(ctx), title="Change orders", icon=":material/rule:", url_path="change-orders"),
    "contractors": st.Page(lambda: PP.contractors(ctx), title="Contractor quality", icon=":material/monitoring:", url_path="contractors"),
    "patterns": st.Page(lambda: (LIM.lab_banner(), PP.patterns(ctx)), title="Patterns", icon=":material/hub:", url_path="patterns"),
    "process": st.Page(lambda: PP.process(ctx), title="Cash & process", icon=":material/account_balance:", url_path="cash-process"),
    "carbon": st.Page(lambda: (LIM.lab_banner(), PP.carbon(ctx)), title="Sustainability", icon=":material/eco:", url_path="carbon"),
    "intake": st.Page(lambda: OP.intake(ctx), title="Data intake", icon=":material/document_scanner:", url_path="intake"),
    "trends": st.Page(lambda: OP.trends(ctx), title="Trends", icon=":material/trending_up:", url_path="trends"),
    "story": st.Page(lambda: OP.story(ctx), title="Story mode", icon=":material/slideshow:", url_path="story"),
    "mobile": st.Page(lambda: OP.mobile(ctx), title="Mobile", icon=":material/smartphone:", url_path="mobile"),
    "google": st.Page(lambda: (LIM.lab_banner(), OP.google_stack(ctx)), title="BigQuery + Looker", icon=":material/cloud:", url_path="bigquery"),
    "analytics": st.Page(analytics, title="Program analytics", icon=":material/analytics:", url_path="analytics"),
    "materials": st.Page(lambda: SCM.page(ctx), title="Materials & supply chain", icon=":material/inventory_2:", url_path="materials"),
    "markets": st.Page(lambda: (LIM.lab_banner(), GM.page(ctx)), title="Global markets & trade", icon=":material/currency_exchange:", url_path="global-markets"),
    "payapp": st.Page(lambda: PAY.payapp_page(ctx), title="Pay applications (G702/G703)", icon=":material/receipt_long:", url_path="pay-applications"),
    "issues": st.Page(lambda: PAY.issues_page(ctx), title="Issues & decisions", icon=":material/fact_check:", url_path="issues"),
    "limits": st.Page(lambda: LIM.page(ctx), title="What this is not (yet)", icon=":material/info:", url_path="limits"),
}

SECTIONS = {
    "Overview": ["front", "home", "insights", "project", "trends", "story"],
    "Bills & contracts": ["issues", "analytics", "payapp", "co", "cost", "contractors"],
    "Predict": ["forecast", "schedule"],
    "Supply": ["materials"],
    "Operations": ["process", "intake", "mobile"],
    "Labs": ["markets", "ml", "patterns", "carbon", "google", "limits"],
}
allowed = ROLES.get(st.session_state.get("role"), None)
nav_dict = {sec: [PAGES[k] for k in keys if allowed is None or k in allowed] for sec, keys in SECTIONS.items()}
nav_dict = {k: v for k, v in nav_dict.items() if v}
nav = st.navigation(nav_dict, position="top")


def _reset_filters():
    for k_ in (f"fp{sk}", f"fc{sk}", f"fm{sk}"):
        st.session_state.pop(k_, None)


parts = []
if len(f_projects) < len(all_projects):
    parts.append(f"{len(f_projects)} of {len(all_projects)} projects")
if len(f_primes) < len(all_primes):
    parts.append(f"{len(f_primes)} of {len(all_primes)} contractors")
if len(f_periods) < len(all_periods):
    parts.append(f"months {f_periods[0]} to {f_periods[-1]}")
if any(tol[k_] != TOLERANCES[k_] for k_ in TOLERANCES) or cap != PER_DIEM_CAP:
    parts.append("custom tolerances")
if parts:
    fb1, fb2 = st.columns([8, 1])
    fb1.markdown(f'<span class="pill warn">Filters applied: {" · ".join(parts)}</span>', unsafe_allow_html=True)
    fb2.button("Reset", key="reset_filters", on_click=_reset_filters, type="tertiary", help="Back to all projects, contractors and months.")
if "_goto" in st.session_state:
    target = st.session_state.pop("_goto")
    if allowed is None or target in allowed:
        st.switch_page(PAGES[target])
nav.run()
