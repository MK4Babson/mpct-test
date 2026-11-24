"""Data intake (Bill Reader, delivery photos, Google Sheets), Trends (monthly snapshots), Story mode + exports,
Mobile view, and the Google stack (BigQuery + Looker Studio) package."""
import io
import zipfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ai
import intake as I
import report as REP
from insights import RULE_LABEL, money
from schema import SCHEMA
from ui.cost_pages import back, hero, howto, synthetic_note
from ui.theme import SERIES, STATUS, GATE_COLOR

ROOT = Path(__file__).resolve().parents[1]
import tempfile  # noqa: E402
SAMPLE_DIR = Path(tempfile.gettempdir()) / "mpct_samples"  # generated samples: writable on any host
GATE_ICON = {"PASS": "🟢 PASS", "REVIEW": "🟠 REVIEW", "HOLD": "🔴 HOLD"}


# ---------------------------------------------------------------- Intake
def intake(ctx):
    back(ctx, "in")
    hero("Data intake · from the field and the contractor", "No more manual uploads",
         "Read PDF pay applications into checked lines, confirm deliveries from a site photo, and pull contractor submissions from "
         "locked Google Sheets with a tamper-evident receipt.")
    st.caption("AI: " + (f"Gemini ({ai.model_name()}) connected for scans and handwriting." if ai.available() else
                         "not connected. PDFs with a text layer and printed delivery notes still work (pdfplumber + Tesseract OCR). "
                         "Add GEMINI_API_KEY in the app's secrets for scanned bills."))
    tabs = st.tabs(["Bill Reader (PDF)", "Site delivery photos", "Google Sheets intake"])
    with tabs[0]:
        _bill_reader(ctx)
    with tabs[1]:
        _delivery(ctx)
    with tabs[2]:
        _sheets(ctx)


@st.cache_data(show_spinner=False)
def _samples(_tables, _gates, key, names, labels):
    folder = SAMPLE_DIR / key
    files = I.build_samples(_tables, _gates, names, labels, folder)
    return [(p.name, p.read_bytes()) for p in files]


def _bill_reader(ctx):
    st.markdown("Upload a contractor's pay application as PDF. The reader pulls the header and every line, then the rules engine "
                "checks it against the PO, field receipts and approvals, exactly as if it had arrived as data.")
    samples = _samples(ctx.tables, ctx.gates, ctx.source_key, ctx.names, ctx.labels)
    c1, c2 = st.columns([2, 3])
    up = c1.file_uploader("Pay application (PDF)", type="pdf", key="bill_pdf")
    pick = c2.radio("…or try a sample bill from this program", ["(none)"] + [n for n, _ in samples], key="bill_sample")
    for n, b in samples:
        c2.download_button(f"Download {n}", b, n, key=f"dl_{n}")
    data = up.getvalue() if up else dict(samples).get(pick)
    if not data:
        return
    hdr, lines, method, warns = I.read_bill_pdf(data)
    st.markdown(f"**Read with:** {method}")
    for w in warns:
        st.warning(w)
    if not len(lines):
        st.error("No lines found.")
        return
    k = st.columns(5)
    k[0].metric("Invoice", hdr.get("invoice_number") or "?")
    k[1].metric("PO", hdr.get("po_id") or "?")
    k[2].metric("Period", hdr.get("period") or "?")
    k[3].metric("Lines read", len(lines))
    k[4].metric("Gross", money(hdr.get("gross_amount") or 0))
    st.dataframe(lines, hide_index=True, width="stretch")
    try:
        t2, iid = I.to_tables(hdr, lines, ctx.tables)
    except ValueError as e:
        st.error(str(e))
        return
    from run_engine import run
    ex, g = run(tables=t2)
    gr = g[g.invoice_id == iid].iloc[0]
    e = ex[ex.invoice_id == iid]
    with st.container(border=True):
        st.markdown(f"### Result: {GATE_ICON[gr.gate]}  ·  {money(gr.amount_at_risk)} at risk")
        if len(e):
            st.dataframe(e.assign(issue=e.rule.map(lambda r: RULE_LABEL.get(r, r)))[["line_no", "issue", "severity", "amount_at_risk", "detail"]],
                         hide_index=True, width="stretch", column_config={"amount_at_risk": st.column_config.NumberColumn("At risk", format="$%,.2f")})
        else:
            st.success("Every line matches the PO, the field receipts and the approvals. Ready to pay.")
    orig = ctx.gates[ctx.gates.invoice_id == iid]
    if len(orig):
        same = orig.gate.iloc[0] == gr.gate and abs(orig.amount_at_risk.iloc[0] - gr.amount_at_risk) < 1
        st.caption(("✓ " if same else "⚠ ") + f"This bill is already in the program data as {iid}; reading it from the PDF gives "
                   + ("the same result as the structured data." if same else "a different result: check the extraction."))


@st.cache_data(show_spinner=False)
def _photo_samples(_tables, key, coords):
    folder = SAMPLE_DIR / key
    return [(p.name, p.read_bytes()) for p in I.build_photo_samples(_tables, dict(coords), folder)]


def _delivery(ctx):
    st.markdown("The site manager photographs the delivery note as material is unloaded. The app reads the note, checks the photo's "
                "GPS and time against the site, matches the PO line, and shows whether that material has been billed and paid.")
    coords = {p: ctx.coords_of(p) for p in ctx.pt.project_id}
    coords = {p: c for p, c in coords.items() if not any(np.isnan(c))}
    if not coords:
        st.info("This program has no site coordinates.")
        return
    samples = _photo_samples(ctx.tables, ctx.source_key, tuple(sorted(coords.items())))
    c1, c2, c3 = st.columns([2, 2, 2])
    up = c1.file_uploader("Photo of the delivery note", type=["jpg", "jpeg", "png"], key="dn_up")
    cam = c2.camera_input("…or take one now", key="dn_cam", label_visibility="visible") if st.session_state.get("dn_cam_on") else None
    c2.toggle("Use the camera", key="dn_cam_on")
    pick = c3.radio("…or try a sample", ["(none)"] + [n for n, _ in samples], key="dn_sample")
    data = (up.getvalue() if up else None) or (cam.getvalue() if cam else None) or dict(samples).get(pick)
    if not data:
        return
    meta = I.read_photo_meta(data)
    fields, method, raw = I.read_photo_text(data)
    c1, c2 = st.columns([2, 3])
    c1.image(data, width="stretch")
    with c2:
        st.markdown(f"**Read with:** {method or 'nothing (no OCR available): type the fields'}")
        f = {}
        cc = st.columns(2)
        f["vehicle"] = cc[0].text_input("Vehicle plate", fields.get("vehicle", ""), key="dn_v")
        f["note_no"] = cc[1].text_input("Delivery note no.", fields.get("note_no", ""), key="dn_n")
        f["po_id"] = cc[0].text_input("PO", fields.get("po_id", ""), key="dn_po")
        f["item_code"] = cc[1].text_input("Item", fields.get("item_code", ""), key="dn_it")
        f["qty"] = cc[0].text_input("Quantity", fields.get("qty", ""), key="dn_q")
        f["supplier"] = cc[1].text_input("Supplier", fields.get("supplier", ""), key="dn_s")
        st.caption(f"Photo metadata: GPS {meta['lat']:.4f}, {meta['lng']:.4f}" if meta["lat"] is not None else "Photo metadata: no GPS")
        if meta["taken"]:
            st.caption(f"Taken {meta['taken']:%d %b %Y %H:%M}")
    res = I.match_delivery(f, meta, ctx.tables, coords)
    with st.container(border=True):
        ok = all(c[1] for c in res["checks"])
        st.markdown(f"### {'🟢 Supporting evidence matches' if ok else '🔴 Needs a look'} · {res['status']}")
        st.caption("A photo is supporting evidence, never proof of delivery: its location data can be missing, edited or reused, and "
                   "it shows neither quantity nor condition. Delivery is confirmed by the delivery note, the authorized receiving "
                   "record and any QA inspection; the photo backs them up.")
        for name, passed, detail in res["checks"]:
            st.markdown(f"{'✓' if passed else '✕'} **{name}:** {detail}")
        if res["project_id"]:
            st.markdown(f"Billed to date on this item: **{res['billed_qty_to_date']:,.0f}** units"
                        + (f"; paid {money(res['paid'])}." if res["paid"] else "."))
    log = st.session_state.setdefault("deliveries", [])
    if st.button("Record this delivery", key="dn_rec"):
        log.append({**f, "project_id": res["project_id"], "distance_km": res["distance_km"], "status": res["status"],
                    "taken": meta["taken"], "confirmed": all(c[1] for c in res["checks"])})
    if log:
        df = pd.DataFrame(log)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download delivery log (CSV)", df.to_csv(index=False), "delivery_log.csv")
    synthetic_note("Sample photos are generated delivery notes with GPS stamped into the file. Real phone photos carry GPS when location is on.")


def _sheets(ctx):
    st.markdown("Each contractor fills a **locked Google Sheet** built from the templates (headers protected, one tab per table). "
                "On submit, an Apps Script freezes the tab and stamps a hash. Here, paste each sheet's link: the app reads it, "
                "validates it, and issues a receipt (SHA-256) so any later edit is detectable.")
    with st.expander("Set up the Google Sheets template (once)"):
        st.markdown("1. Create a Google Sheet with one tab per table: " + ", ".join(f"`{t}`" for t in SCHEMA) + ".\n"
                    "2. Paste the headers from the blank templates (sidebar → Upload → Download blank templates).\n"
                    "3. Extensions → Apps Script → paste the script below → run `setup()` once. It protects the header rows and "
                    "adds a **Submit** menu that freezes the tab and records a hash.\n"
                    "4. Share → 'Anyone with the link can view' (or publish to the web) so the app can read it.")
        st.code((ROOT / "templates" / "sheets_intake" / "Code.gs").read_text() if (ROOT / "templates" / "sheets_intake" / "Code.gs").exists() else "",
                language="javascript")
    rows = []
    c = st.columns(2)
    for i, t in enumerate(["invoices", "invoice_lines", "goods_receipts", "change_orders"]):
        url = c[i % 2].text_input(f"{t} sheet link", key=f"gs_{t}", placeholder="https://docs.google.com/spreadsheets/d/…/edit#gid=0")
        if url:
            df, err = I.read_sheet(url)
            if err:
                st.error(f"{t}: {err}")
            else:
                missing = [col for col, *_ in SCHEMA[t] if col not in df.columns]
                rows.append({**I.freeze(df, t), "missing_columns": ", ".join(missing) or "none"})
                st.session_state.setdefault("sheet_tables", {})[t] = df
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.download_button("Download submission receipts (JSON)", I.receipt_json(rows), "submission_receipts.json")
        st.caption("Tables read here replace the same tables of the current program when you choose 'Upload my own CSV files' and "
                   "export them; a direct sheet-to-engine run is on the roadmap.")
    else:
        st.caption("No sheet linked yet. Try any public sheet with the template headers.")


# ---------------------------------------------------------------- Trends
def trends(ctx):
    back(ctx, "tr")
    hero("Trends · is the program improving?", "Month by month", "Each point is the program as it stood at that month-end: the same "
         "calculations re-run on the data available then. Watch the direction, not one month.")
    snap = ctx.monthly()
    if not len(snap) or len(snap) < 2:
        st.info("Needs at least two months of data.")
        return
    last, prev = snap.iloc[-1], snap.iloc[-2]
    k = st.columns(5)
    k[0].metric("Program health", f"{last.health:.0f}", f"{last.health - prev.health:+.0f} vs last month")
    k[1].metric("First-pass rate", f"{last.first_pass:.0f}%", f"{last.first_pass - prev.first_pass:+.0f} pts")
    k[2].metric("Flagged (recommend hold)", money(last.blocked), money(last.blocked - prev.blocked), delta_color="inverse")
    k[3].metric("Verified vs plan", f"{last.ver_minus_plan:+.0f} pts", f"{last.ver_minus_plan - prev.ver_minus_plan:+.0f}")
    k[4].metric("Forecast vs budget", money(last.vac), money(last.vac - prev.vac))
    metrics = {"health": "Program health (0-100)", "first_pass": "First-pass rate (%)", "blocked_m": "Blocked ($M)",
               "ver_minus_plan": "Verified minus plan (pts)", "vac_m": "Under (+) / over (−) budget ($M)", "at_risk_sites": "Projects at risk"}
    snap = snap.assign(blocked_m=snap.blocked / 1e6, vac_m=snap.vac / 1e6)
    cols = st.columns(3)
    for i, (m, lab) in enumerate(metrics.items()):
        ch = alt.Chart(snap).mark_line(color=SERIES[i % len(SERIES)], strokeWidth=2, point=alt.OverlayMarkDef(size=50, filled=True)).encode(
            x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y(f"{m}:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=["period", alt.Tooltip(f"{m}:Q", format=",.2f", title=lab)]).properties(height=170, title=lab)
        cols[i % 3].altair_chart(ch, width="stretch")
    best = "improving" if last.first_pass > snap.first_pass.iloc[: max(1, len(snap) // 2)].mean() else "not improving"
    howto(f"<b>Read it:</b> first-pass rate is {best} versus the first half of the program. Blocked money and the budget gap show "
          "whether problems are being closed or piling up. Snapshots are kept here monthly because the data is monthly; with weekly "
          "data the same code stores weekly snapshots.")
    st.download_button("Download snapshots (CSV)", snap.to_csv(index=False), "monthly_snapshots.csv")


# ---------------------------------------------------------------- Story
def story(ctx):
    back(ctx, "st")
    hero("Story mode · for leadership", "The program in five minutes", "Five chapters, one message each: where we stand, where money "
         "is at risk, why, what it will cost, and the decisions needed. Export as PDF or PowerPoint.")
    chapters = REP.build_story(ctx.program_label, ctx.pt, ctx.prog, ctx.gf, ctx.ef, ctx.risk, ctx.rc, ctx.perf, ctx.insights, ctx.health,
                               ctx.labels, ctx.end_period)
    i = st.session_state.get("story_i", 0)
    i = min(i, len(chapters) - 1)
    nav = st.columns([1, 6, 1])
    if nav[0].button("← Previous", disabled=i == 0, key="st_prev"):
        st.session_state["story_i"] = i - 1
        st.rerun()
    if nav[2].button("Next →", disabled=i == len(chapters) - 1, key="st_next"):
        st.session_state["story_i"] = i + 1
        st.rerun()
    nav[1].progress((i + 1) / len(chapters), text=f"Chapter {i + 1} of {len(chapters)}")
    c = chapters[i]
    with st.container(border=True):
        st.markdown(f'<div class="eyebrow">{c["title"]}</div><div class="hero-title" style="font-size:2rem">{c["headline"]}</div>', unsafe_allow_html=True)
        cc = st.columns([2, 3]) if c["chart"] is not None else [st.container(), None]
        with cc[0]:
            for p in c["points"]:
                if p:
                    st.markdown(f"- {p}")
        if c["chart"] is not None and cc[1] is not None:
            kind, df = c["chart"]
            with cc[1]:
                if kind == "lines" and len(df):
                    long = df.melt(id_vars="period", var_name="series", value_name="pct").dropna()
                    dom = ["Planned", "Verified", "Billed"]
                    st.altair_chart(alt.Chart(long).mark_line(strokeWidth=2, point=True).encode(
                        x=alt.X("period:O", title=None), y=alt.Y("pct:Q", title="% complete", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=[SERIES[3], SERIES[2], SERIES[0]]), legend=alt.Legend(orient="top", title=None)),
                        strokeDash=alt.condition(alt.datum.series == "Planned", alt.value([5, 4]), alt.value([1, 0]))).properties(height=280), width="stretch")
                elif kind == "barh" and len(df):
                    st.altair_chart(alt.Chart(df).mark_bar(color=SERIES[1], height=18, cornerRadiusEnd=4).encode(
                        y=alt.Y("label:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)), x=alt.X("value:Q", title=None)).properties(height=alt.Step(30)),
                        width="stretch")
                elif kind == "pair" and len(df):
                    long = df.melt(id_vars="name", value_vars=["budget", "eac"], var_name="m", value_name="v")
                    long["m"] = long.m.map({"budget": "Budget", "eac": "Forecast"})
                    st.altair_chart(alt.Chart(long).mark_bar(height=12, cornerRadiusEnd=3).encode(
                        y=alt.Y("name:N", title=None), yOffset="m:N", x=alt.X("v:Q", title="$", axis=alt.Axis(format="$~s")),
                        color=alt.Color("m:N", scale=alt.Scale(domain=["Budget", "Forecast"], range=[SERIES[0], SERIES[1]]), legend=alt.Legend(orient="top", title=None))
                    ).properties(height=alt.Step(14)), width="stretch")
    st.markdown("#### Export")
    c1, c2 = st.columns(2)
    if c1.button("Build PDF brief", key="st_pdf"):
        st.session_state["story_pdf"] = REP.story_pdf(chapters, ctx.program_label)
    if c2.button("Build PowerPoint", key="st_ppt"):
        st.session_state["story_ppt"] = REP.story_pptx(chapters, ctx.program_label)
    if st.session_state.get("story_pdf"):
        c1.download_button("Download PDF", st.session_state["story_pdf"], "executive_brief.pdf", "application/pdf")
    if st.session_state.get("story_ppt"):
        c2.download_button("Download PowerPoint", st.session_state["story_ppt"], "executive_brief.pptx",
                           "application/vnd.openxmlformats-officedocument.presentationml.presentation")


# ---------------------------------------------------------------- Mobile
MOBILE_CSS = """<style>
.m-card { background: rgba(14,21,37,.75); border:1px solid rgba(120,170,255,.18); border-radius:14px; padding:12px 14px; margin:8px 0; }
.m-big { font-family:'JetBrains Mono',monospace; font-size:2.2rem; color:#e8eef8; line-height:1; }
.m-k { font-size:.7rem; color:#8792a6; text-transform:uppercase; letter-spacing:.08em; }
.m-row { display:flex; gap:10px; } .m-row > div { flex:1; }
</style>"""


def mobile(ctx):
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Mobile · decide on the go</div><div class="hero-title" style="font-size:1.8rem">Today</div>', unsafe_allow_html=True)
    st.caption("A single-column view sized for a phone. On Android or iPhone, open the app link in Chrome or Safari and use "
               "'Add to Home screen' to install it like an app.")
    pt, H = ctx.pt, ctx.health
    w = pt.set_index("project_id").committed
    hl = (H.set_index("project_id").health.fillna(0) * w).sum() / w.sum()
    st.markdown(f'<div class="m-card"><div class="m-row"><div><div class="m-k">Program health</div><div class="m-big">{hl:.0f}</div></div>'
                f'<div><div class="m-k">Blocked</div><div class="m-big" style="font-size:1.5rem">{money(ctx.risk.blocked.sum())}</div></div>'
                f'<div><div class="m-k">At risk sites</div><div class="m-big" style="font-size:1.5rem">{int((H.health_status == "At risk").sum())}</div></div></div></div>',
                unsafe_allow_html=True)
    log = st.session_state.setdefault("decision_log", [])
    st.markdown("##### Bills waiting for you")
    g = ctx.gf[ctx.gf.gate != "PASS"].sort_values("amount_at_risk", ascending=False).head(8)
    done = {d["item"] for d in log}
    for r in g.itertuples():
        if r.invoice_id in done:
            continue
        with st.container(border=True):
            st.markdown(f"**{GATE_ICON[r.gate]} · {r.invoice_number}** · {ctx.labels.get(r.project_id, r.project_id)}  \n"
                        f"{ctx.names.get(r.prime_id, r.prime_id)} · {money(r.gross_amount)} billed · **{money(r.amount_at_risk)} at risk**  \n"
                        f":gray[{'; '.join(RULE_LABEL.get(x.strip(), x) for x in str(r.rules_triggered).split(';') if x.strip())}]")
            b = st.columns(3)
            for col, act in zip(b, ["Pay verified only", "Hold", "Ask contractor"]):
                if col.button(act, key=f"m_{r.invoice_id}_{act}", width="stretch"):
                    log.append({"item": r.invoice_id, "invoice": r.invoice_number, "action": act, "at_risk": r.amount_at_risk,
                                "when": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")})
                    st.rerun()
    st.markdown("##### Alerts")
    for d in ctx.insights[:5]:
        icon = {"critical": "🔴", "warning": "🟠", "info": "🔵", "good": "🟢"}[d["level"]]
        st.markdown(f'<div class="m-card">{icon} <b>{d["title"]}</b><br><span style="color:#b9c3d3;font-size:.85rem">{d["action"]}</span></div>',
                    unsafe_allow_html=True)
    if log:
        st.markdown("##### Your decisions")
        df = pd.DataFrame(log)
        st.dataframe(df.drop(columns=["item"]), hide_index=True, width="stretch")
        st.download_button("Send decisions (CSV)", df.to_csv(index=False), "decisions.csv", width="stretch")
        st.caption("Decisions are recorded for the reviewer team; the rules engine still controls what is paid.")


# ---------------------------------------------------------------- Google stack
def google_stack(ctx):
    back(ctx, "gc")
    hero("Google stack · BigQuery + Looker Studio", "Run the same rules in BigQuery",
         "The rules engine is plain SQL. This package translates it to BigQuery Standard SQL, loads the tables, runs the checks and "
         "creates reporting views for a Looker Studio report.")
    folder = ROOT / "bigquery"
    if not folder.exists():
        st.error("bigquery/ folder missing: run `python src/bigquery_export.py`.")
        return
    k = st.columns(4)
    k[0].metric("Rules translated", 11)
    k[1].metric("Reporting views", 3)
    k[2].metric("Table schemas", len(list((folder / "schemas").glob("*.json"))))
    k[3].metric("Round-trip check", "identical", help="The BigQuery SQL was transpiled back to DuckDB and run on the demo data: same gates and same $ at risk as the app.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(folder.parent))
        for t in SCHEMA:
            f = ROOT / "data" / "raw" / f"{t}.csv"
            if f.exists():
                z.write(f, f"bigquery/demo_data/{t}.csv")
    st.download_button("Download the BigQuery package (zip)", buf.getvalue(), "control_tower_bigquery.zip", type="primary")
    t1, t2, t3, t4 = st.tabs(["How to run", "Engine SQL (BigQuery)", "Reporting views", "Load script"])
    t1.markdown((folder / "LOOKER_STUDIO.md").read_text())
    t2.code((folder / "engine_bigquery.sql").read_text(), language="sql")
    t3.code((folder / "reporting_views.sql").read_text(), language="sql")
    t4.code((folder / "load_and_run.sh").read_text(), language="bash")
    howto("<b>Why it matters:</b> the app proves the logic; BigQuery runs it at scale on the owner's real AP and project data, and "
          "Looker Studio shares it with finance and construction teams who already use Google Workspace. Not yet run on a live "
          "BigQuery project: the SQL was validated with sqlglot's BigQuery parser and a DuckDB round trip.")
