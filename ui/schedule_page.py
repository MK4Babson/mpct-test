"""Schedule (CPM / PERT): critical path, float, PERT odds, Monte Carlo schedule risk, crashing, time impact analysis, quality."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import schedule as S
from insights import money
from ui.cost_pages import back, hero, howto, synthetic_note
from ui.theme import SERIES, STATUS

CRIT, NONCRIT, DONE, BASE = STATUS["critical"], SERIES[0], "#3b4a66", "#8792a6"


def _controls_row(ctx, pid):
    pc = ctx.controls.get("project_controls") if ctx.controls else None
    if pc is None or not len(pc) or pid not in set(pc.project_id):
        return None
    r = pc.set_index("project_id").loc[pid].copy()
    if pd.isna(r.get("planned_start")) or pd.isna(r.get("planned_finish")):
        return None
    return r


def _structure(ctx, pid):
    s = ctx.structure_of(pid)
    return s if s in S.TEMPLATES else "datacenter"


@st.cache_data(show_spinner=False)
def _network(_tables, _row, key, pid, end_period, structure):
    return S.build(pid, _tables, _row, end_period, structure)


def network(ctx, pid):
    r = _controls_row(ctx, pid)
    if r is None:
        return None, None
    return _network(ctx.tables, r, ctx.source_key, pid, ctx.end_period, _structure(ctx, pid))


@st.cache_data(show_spinner="Simulating 3,000 schedules...")
def _mc(_t, _meta, key, pid, end_period, n=3000):
    return S.monte_carlo(_t, _meta, n)


@st.cache_data(show_spinner="Working out the cheapest days to buy back...")
def _crash(_t, _meta, key, pid, end_period, days, cost):
    return S.crash(_t, _meta, days, cost)


def co_delay_fn(ctx):
    """For the change-order page: real delay = time impact analysis on the task the change order's item belongs to."""
    cache = {}

    def fn(pid, item, claimed):
        task = S.task_for_item(item) if item else None
        if task is None:
            return claimed, "as claimed (no schedule link)"
        if pid not in cache:
            cache[pid] = network(ctx, pid)
        t, meta = cache[pid]
        if t is None or task not in set(t.id):
            return claimed, "as claimed (no schedule link)"
        d = S.tia(t, meta, task, claimed)
        name = t.set_index("id").name[task]
        return int(round(d)), f"{claimed} days on '{name}' → {d:.0f} on the finish (float absorbs the rest)" if d < claimed - 0.5 else f"{claimed} days on '{name}', on the critical path"
    return fn


def page(ctx):
    back(ctx, "sc")
    hero("Schedule · CPM and PERT", "Which tasks decide the finish date, and how sure are we?",
         "Each building is a network of tasks. The critical path is the longest chain: any delay on it delays the whole job. "
         "Progress comes from the same field-verified quantities used to check the bills.")
    pids = [p for p in ctx.pt.project_id if _controls_row(ctx, p) is not None]
    if not pids:
        st.info("Needs project_controls.csv with planned_start and planned_finish for each project.")
        return
    # program summary
    rows = []
    for p in pids:
        t, meta = network(ctx, p)
        f = S.forecast(t, meta)
        pr = S.pert(t, meta)
        late = f.attrs["finish"] - meta["contract_days"]
        es_fc = ctx.pt.set_index("project_id").forecast_finish.get(p) if "forecast_finish" in ctx.pt else pd.NaT
        rows.append({"project_id": p, "Project": ctx.labels.get(p, p), "Contract finish": meta["planned_finish"].date(),
                     "CPM forecast finish": S.dates(meta, [f.attrs["finish"]])[0].date(), "Days late (CPM)": round(late),
                     "Earned-schedule forecast": pd.Timestamp(es_fc).date() if pd.notna(es_fc) else None,
                     "PERT chance on time": pr["p_on_time"], "Critical path via": " → ".join(f[f.critical].sort_values("es").name.str.split(r"[,:(]", regex=True).str[0].str.strip().head(4))})
    summ = pd.DataFrame(rows)
    st.dataframe(summ.drop(columns="project_id"), hide_index=True, width="stretch", column_config={
        "PERT chance on time": st.column_config.ProgressColumn("PERT chance on time", min_value=0, max_value=1, format="percent")})
    worst = summ.sort_values("Days late (CPM)", ascending=False).project_id.iloc[0]
    pid = st.selectbox("Project", pids, index=pids.index(worst), format_func=lambda p: ctx.labels.get(p, p), key="sch_pid")
    t, meta = network(ctx, pid)
    f = S.forecast(t, meta)
    pr = S.pert(t, meta)
    mc = _mc(t, meta, ctx.source_key, pid, ctx.end_period)
    late = f.attrs["finish"] - meta["contract_days"]
    k = st.columns(5)
    k[0].metric("CPM forecast finish", f"{S.dates(meta, [f.attrs['finish']])[0]:%d %b %Y}", f"{late:+.0f} days vs contract", delta_color="inverse",
                help="Remaining work re-scheduled from the data date through the task network, using most-likely durations.")
    k[1].metric("Critical tasks left", int((f.critical & ~f.done).sum()), help="Unfinished tasks on the longest chain: a day lost on any of them is a day lost at the end.")
    k[2].metric("PERT chance on time", f"{pr['p_on_time']:.0%}", help="Normal approximation on the critical path with three-point estimates.")
    k[3].metric("Monte Carlo P80 finish", f"{S.dates(meta, [mc['p80']])[0]:%d %b %Y}", f"chance on time {mc['p_on_time']:.0%}", delta_color="off", delta_arrow="off")
    k[4].metric("Merge bias", f"{mc['merge_bias']:+.1f} days", help="Extra delay the simulation finds beyond PERT, because several paths must all finish before a merge point.")
    tabs = st.tabs(["Critical path (CPM)", "PERT", "Schedule risk (Monte Carlo)", "Crashing", "Change-order delays (TIA)", "Schedule quality"])

    # ---------- CPM
    with tabs[0]:
        g = f.copy()
        g["Task"] = g.id + "  " + g.name
        g["start_d"] = S.dates(meta, g.fstart.fillna(g.bl_start))
        g["finish_d"] = S.dates(meta, g.ffinish.fillna(g.bl_finish))
        g["bl_s"], g["bl_f"] = S.dates(meta, g.bl_start), S.dates(meta, g.bl_finish)
        g["float_end"] = S.dates(meta, (g.ffinish + g.total_float.clip(lower=0)).fillna(g.bl_finish))
        g["status"] = np.where(g.done, "Done", np.where(g.critical, "Critical", "Has float"))
        order = list(g.sort_values(["bl_start", "id"]).Task)
        y = alt.Y("Task:N", sort=order, title=None, axis=alt.Axis(labelLimit=330))
        base = alt.Chart(g).mark_bar(height=4, color=BASE, opacity=0.6).encode(y=y, x=alt.X("bl_s:T", title=None), x2="bl_f:T")
        flt = alt.Chart(g[~g.done & (g.total_float > 0)]).mark_bar(height=12, color=NONCRIT, opacity=0.18).encode(y=y, x="finish_d:T", x2="float_end:T")
        bars = alt.Chart(g).mark_bar(height=12, cornerRadius=3).encode(
            y=y, x="start_d:T", x2="finish_d:T",
            color=alt.Color("status:N", scale=alt.Scale(domain=["Critical", "Has float", "Done"], range=[CRIT, NONCRIT, DONE]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["Task", "status", alt.Tooltip("pct:Q", format=".0f", title="% complete"), "pct_source",
                     alt.Tooltip("start_d:T", title="Forecast start"), alt.Tooltip("finish_d:T", title="Forecast finish"),
                     alt.Tooltip("total_float:Q", format=".0f", title="Total float (days)"), alt.Tooltip("free_float:Q", format=".0f", title="Free float (days)")])
        lines = pd.DataFrame({"d": [meta["data_date"], meta["planned_finish"]], "what": ["Data date", "Contract finish"]})
        rl = alt.Chart(lines).mark_rule(strokeDash=[4, 3], strokeWidth=2).encode(x="d:T", color=alt.Color("what:N", scale=alt.Scale(domain=["Data date", "Contract finish"], range=["#e8eef8", STATUS["warning"]]), legend=alt.Legend(orient="top", title=None)))
        st.altair_chart(alt.layer(base, flt, bars, rl).resolve_scale(color="independent").properties(height=alt.Step(20)), width="stretch")
        cp = f[f.critical & ~f.done].sort_values("es")
        howto(f"<b>How to read:</b> red bars are the critical path; blue bars have float (the faint extension shows how far they can slip "
              f"before they delay the finish); the thin grey line under each bar is the original plan. <b>Now:</b> the remaining critical path runs "
              f"{' → '.join(cp.name.head(5))}{' …' if len(cp) > 5 else ''}, finishing {abs(late):.0f} days {'after' if late > 0 else 'before'} the contract date. "
              "<b>Decisions:</b> put supervision, overtime and expediting on red tasks only; a slip on a blue task within its float costs nothing.")
        with st.expander("Network diagram"):
            dot = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded", fontname="Helvetica", fontsize=10, fontcolor="#e8eef8"]; edge [color="#5a6b8c"];']
            for r in f.itertuples():
                col = DONE if r.done else (CRIT if r.critical else NONCRIT)
                dot.append(f'{r.id} [label="{r.id}\\n{r.name[:28]}\\nfloat {r.total_float:.0f} d", color="{col}", penwidth={2.5 if r.critical else 1}];')
                for p in r.preds:
                    crit_edge = r.critical and f.set_index("id").critical[p]
                    dot.append(f'{p} -> {r.id} [color="{CRIT if crit_edge else "#5a6b8c"}", penwidth={2 if crit_edge else 1}];')
            dot.append("}")
            st.graphviz_chart("\n".join(dot), width="stretch")
        tbl = f[["id", "name", "trade", "pct", "pct_source", "dur", "remaining", "fstart", "ffinish", "total_float", "free_float", "critical"]].copy()
        tbl["fstart"], tbl["ffinish"] = S.dates(meta, tbl.fstart).date, S.dates(meta, tbl.ffinish).date
        st.dataframe(tbl, hide_index=True, width="stretch", column_config={
            "id": "ID", "name": "Task", "trade": "Trade", "pct": st.column_config.NumberColumn("% done", format="%.0f"), "pct_source": "Progress from",
            "dur": st.column_config.NumberColumn("Duration (d)", format="%.0f"), "remaining": st.column_config.NumberColumn("Remaining (d)", format="%.0f"),
            "fstart": "Forecast start", "ffinish": "Forecast finish", "total_float": st.column_config.NumberColumn("Total float (d)", format="%.0f"),
            "free_float": st.column_config.NumberColumn("Free float (d)", format="%.0f"), "critical": "Critical"})
        st.caption("Total float is measured against the contract finish date, so negative float means the task is already behind it. "
                   "Free float = days a task can slip without delaying the next task.")

    # ---------- PERT
    with tabs[1]:
        pt_ = pr["table"]
        cpt = pt_[pt_.critical & (pt_.te > 0)].sort_values("es")[["name", "o", "m", "p", "te", "sd", "pct"]]
        st.markdown("Each task gets three estimates: optimistic (o), most likely (m) and pessimistic (p). PERT expects "
                    "te = (o + 4m + p) / 6 days, with spread sd = (p − o) / 6, scaled by the work still to do.")
        st.dataframe(cpt, hide_index=True, width="stretch", column_config={
            "name": "Critical task", "o": st.column_config.NumberColumn("Optimistic", format="%.0f"), "m": st.column_config.NumberColumn("Most likely", format="%.0f"),
            "p": st.column_config.NumberColumn("Pessimistic", format="%.0f"), "te": st.column_config.NumberColumn("Expected (remaining)", format="%.1f"),
            "sd": st.column_config.NumberColumn("Spread sd", format="%.1f"), "pct": st.column_config.NumberColumn("% done", format="%.0f")})
        lo, hi = pr["te_finish"] - 3.5 * max(pr["sd"], 1), pr["te_finish"] + 3.5 * max(pr["sd"], 1)
        xs = np.linspace(lo, hi, 140)
        from scipy import stats as _st
        dens = pd.DataFrame({"date": S.dates(meta, xs), "density": _st.norm.pdf(xs, pr["te_finish"], max(pr["sd"], 1e-6))})
        target = st.date_input("Chance of finishing by", value=meta["planned_finish"].date(), key=f"pert_by_{pid}")
        day = (pd.Timestamp(target) - meta["start"]).days
        pb = S.p_by(pr, day)
        area = alt.Chart(dens).mark_area(opacity=0.35, color=SERIES[0]).encode(x=alt.X("date:T", title="Finish date"), y=alt.Y("density:Q", title=None, axis=None))
        shade = alt.Chart(dens[dens.date <= pd.Timestamp(target)]).mark_area(opacity=0.6, color=SERIES[2]).encode(x="date:T", y="density:Q")
        rl = alt.Chart(pd.DataFrame({"d": [pd.Timestamp(target)]})).mark_rule(color=STATUS["warning"], strokeDash=[4, 3], strokeWidth=2).encode(x="d:T")
        st.altair_chart((area + shade + rl).properties(height=220), width="stretch")
        st.metric(f"Chance of finishing by {pd.Timestamp(target):%d %b %Y}", f"{pb:.0%}",
                  help=f"Expected finish {S.dates(meta, [pr['te_finish']])[0]:%d %b %Y}, spread ±{pr['sd']:.0f} days (1 sd) along the critical path.")
        howto("<b>How to read:</b> the curve is the spread of likely finish dates; the green area is the chance of finishing by the date you pick. "
              "Pessimistic estimates are further from 'most likely' than optimistic ones, so the expected finish sits later than the plan. "
              "<b>Limit:</b> PERT looks only at the one critical path; the Monte Carlo tab adds the paths that could overtake it.")

    # ---------- Monte Carlo
    with tabs[2]:
        fin = pd.DataFrame({"date": S.dates(meta, mc["finish"])})
        h = alt.Chart(fin).mark_bar(color=SERIES[0], opacity=0.85).encode(x=alt.X("date:T", bin=alt.Bin(maxbins=40), title="Finish date"), y=alt.Y("count():Q", title="Simulations"))
        refs = pd.DataFrame({"d": S.dates(meta, [mc["p80"], meta["contract_days"]]), "what": ["P80", "Contract finish"]})
        rr = alt.Chart(refs).mark_rule(strokeDash=[4, 3], strokeWidth=2).encode(x="d:T", color=alt.Color("what:N", scale=alt.Scale(domain=["P80", "Contract finish"], range=[CRIT, STATUS["warning"]]), legend=alt.Legend(orient="top", title=None)))
        c1, c2 = st.columns([3, 2])
        c1.altair_chart((h + rr).properties(height=280, title="3,000 simulated finish dates"), width="stretch")
        ci = mc["criticality"].sort_values(ascending=False).head(12).reset_index()
        ci.columns = ["id", "ci"]
        ci["Task"] = ci.id + " " + ci.id.map(t.set_index("id").name)
        c2.altair_chart(alt.Chart(ci).mark_bar(height=12, cornerRadiusEnd=3, color=CRIT).encode(
            y=alt.Y("Task:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("ci:Q", title="Criticality index", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            tooltip=["Task", alt.Tooltip("ci:Q", format=".0%")]).properties(height=alt.Step(20), title="How often each task is critical"), width="stretch")
        near = ci[(ci.ci > 0.1) & (ci.ci < 0.9)]
        howto("<b>How it works:</b> every task's remaining duration is drawn from its three-point (beta-PERT) range, the network is re-run "
              f"3,000 times, and each run's finish and critical path are recorded. <b>Now:</b> P50 {S.dates(meta, [mc['p50']])[0]:%d %b}, P80 "
              f"{S.dates(meta, [mc['p80']])[0]:%d %b}, chance on time {mc['p_on_time']:.0%}. "
              + (f"Near-critical tasks ({', '.join(near.Task.str.split(' ', n=1).str[1].head(3))}) become critical in some futures: watch them too. " if len(near) else
                 "One path dominates: the same tasks are critical in almost every run. ")
              + f"Merge bias {mc['merge_bias']:+.1f} days: where paths join, the latest one wins, so single-path PERT is optimistic.")

    # ---------- Crashing
    with tabs[3]:
        c = st.columns(3)
        default_cost = 40000.0
        dc = c[0].number_input("Owner's cost of one day of delay ($)", 0.0, 1e6, default_cost, 5000.0, key=f"cr_dc_{pid}",
                               help="Lost revenue, extended site costs, rented power. Same assumption as on the Change orders page.")
        want = c[1].slider("Days to try to buy back", 1, 40, int(min(40, max(5, round(late)))), key=f"cr_days_{pid}")
        cr = _crash(t, meta, ctx.source_key, pid, ctx.end_period, want, dc)
        if not len(cr):
            st.info("Nothing left to speed up on the critical path.")
        else:
            worth = cr[cr.worth_it]
            k = st.columns(4)
            k[0].metric("Days bought back", f"{cr.days_saved_total.iloc[-1]:.0f}")
            k[1].metric("Cost", money(cr.cumulative_cost.iloc[-1]))
            k[2].metric("Worth buying (cost per day < delay cost)", f"{worth.days_saved_total.max() if len(worth) else 0:.0f} days")
            k[3].metric("Net saving on those days", money((worth.days_saved * dc).sum() - worth.cost.sum()) if len(worth) else "$0")
            cr["delay_avoided"] = cr.days_saved_total * dc
            long = pd.concat([cr.assign(series="Cost of crashing", v=cr.cumulative_cost), cr.assign(series="Delay cost avoided", v=cr.delay_avoided)])
            ch = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
                x=alt.X("days_saved_total:Q", title="Days bought back"), y=alt.Y("v:Q", title="$", axis=alt.Axis(format="$~s")),
                color=alt.Color("series:N", scale=alt.Scale(domain=["Cost of crashing", "Delay cost avoided"], range=[SERIES[1], SERIES[2]]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["series", alt.Tooltip("days_saved_total:Q", format=".1f"), alt.Tooltip("v:Q", format="$,.0f")])
            st.altair_chart(ch.properties(height=250), width="stretch")
            st.dataframe(cr[["step", "task", "cost", "days_saved", "cost_per_day", "cumulative_cost", "worth_it"]], hide_index=True, width="stretch", column_config={
                "step": "Step", "task": "Speed up", "cost": st.column_config.NumberColumn("Cost", format="$%,.0f"), "days_saved": st.column_config.NumberColumn("Days saved", format="%.1f"),
                "cost_per_day": st.column_config.NumberColumn("$ per day saved", format="$%,.0f"), "cumulative_cost": st.column_config.NumberColumn("Total", format="$%,.0f"),
                "worth_it": "Cheaper than delay?"})
            howto("<b>How it works:</b> at each step the app finds the cheapest way to take one day off the finish: one critical task, or one "
                  "task on each of two parallel critical paths (crashing only one of them saves nothing). It stops when no critical task can "
                  "be shortened further. <b>Use it:</b> buy days while the green line rises faster than the orange one; after that, accept the delay.")
        synthetic_note("Crash costs per day and limits are illustrative (overtime, extra crews, air freight for long-lead equipment).")

    # ---------- TIA
    with tabs[4]:
        st.markdown("**Time impact analysis:** insert the delay a change would cause into the schedule, re-run it, and read how much the finish "
                    "moves. A delay on a task with float can cost nothing; on the critical path it costs every day.")
        st.caption("Screening only. This shows whether a claimed delay is even plausible on a template schedule. A real delay "
                   "entitlement needs the contractor's accepted baseline and updates, concurrent delays, notice terms and expert review "
                   "(AACE practice); a finish-date shift here is not compensable days.")
        det = ctx.extras.get("co_details")
        co = ctx.tables["change_orders"].copy()
        if "project_id" not in co.columns:
            co["project_id"] = co.po_id.map(ctx.tables["purchase_orders"].set_index("po_id").project_id)
        co = co[(co.project_id == pid) & (co.status == "PENDING")]
        if det is not None and len(co):
            co = co.merge(det[["co_id", "schedule_days_if_rejected"]], on="co_id", how="left")
            rows = []
            for r in co.itertuples():
                item = getattr(r, "base_item_code", None)
                task = S.task_for_item(item) if item else None
                claimed = int(r.schedule_days_if_rejected or 0)
                if task and task in set(t.id) and claimed:
                    real = S.tia(t, meta, task, claimed)
                    fl = f.set_index("id").total_float[task]
                else:
                    real, fl = (claimed if claimed else 0), np.nan
                rows.append({"co_id": r.co_id, "amount": r.co_amount, "task": t.set_index("id").name.get(task, "not linked") if task else "not linked",
                             "float": fl, "claimed": claimed, "real": real})
            tia = pd.DataFrame(rows)
            st.dataframe(tia, hide_index=True, width="stretch", column_config={
                "co_id": "Change order", "amount": st.column_config.NumberColumn("Amount", format="$%,.0f"), "task": "Schedule task",
                "float": st.column_config.NumberColumn("Task float (d)", format="%.0f"), "claimed": "Delay if rejected (claimed, d)",
                "real": st.column_config.NumberColumn("Delay to the finish (TIA, d)", format="%.0f")})
            if len(tia):
                st.caption("On the Change orders page, turn on 'Use the schedule (time impact analysis)' to price delay with the TIA days instead of the claimed days.")
        else:
            st.caption("No pending change orders on this project.")
        c = st.columns(2)
        task = c[0].selectbox("What if this task slips…", [a for a in f.id if not f.set_index("id").done[a] and f.set_index("id").kind[a] != "milestone"],
                              format_func=lambda a: f"{a} {t.set_index('id').name[a]}", key=f"tia_task_{pid}")
        days = c[1].slider("…by this many days", 1, 60, 10, key=f"tia_days_{pid}")
        moved = S.tia(t, meta, task, days)
        st.metric("Finish moves by", f"{moved:.0f} days", f"float absorbed {days - moved:.0f} days" if moved < days else "on the critical path", delta_color="off", delta_arrow="off")

    # ---------- quality
    with tabs[5]:
        q = S.quality(t, f, meta)
        q["result"] = np.where(q["pass"], "✓ pass", "✕ check")
        st.dataframe(q[["check", "result", "detail"]], hide_index=True, width="stretch")
        howto("<b>Why it matters:</b> a schedule is only as good as its logic. Open ends hide delays, very long tasks cannot be tracked, and "
              "large float often means a missing link. These checks follow the spirit of the US DCMA 14-point assessment used on big programs.")
    synthetic_note("Task networks are templates for a data center building and a canal division, fitted to each contract's planned dates. "
                   f"Progress for {int((t.pct_source == 'field-verified quantities').sum())} of {len(t)} tasks comes from field-verified quantities; "
                   "the rest (long-lead deliveries, the shell by others) are assumed on plan. Load a P6 or MS Project export to use the real network.")
