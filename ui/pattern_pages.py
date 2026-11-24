"""Contractor quality (control charts, segments), Patterns (issue rules, lead-lag, learning curve),
Cash & process (cycle time, seasonal cash forecast), Sustainability (embodied carbon)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import patterns as PA
import process as PR
from insights import RULE_LABEL, money
from ui.cost_pages import back, hero, howto, synthetic_note
from ui.theme import SERIES, STATUS


# ---------------------------------------------------------------- Contractor quality
def contractors(ctx):
    back(ctx, "cq")
    hero("Contractor quality · statistical process control", "Is a contractor really getting worse?",
         "Control charts separate a bad month from a real change in a contractor's billing quality. Segments group contract "
         "packages that behave alike, so each gets a review policy that fits.")
    pc, verdict = PA.p_chart(ctx.tables, ctx.gf, ctx.ef, ctx.names)
    if not len(pc):
        st.info("No bills in this view.")
        return
    st.markdown("#### Control charts (p-chart: share of bill lines with an issue)")
    for v in verdict.itertuples():
        st.markdown(("🔴 " if "worse" in v.signal else "🟢 ") + f"**{v.contractor}:** {v.signal} ({v.evidence}). "
                    f"Average {v.average_issue_rate:.1%} of lines; latest month {v.latest:.1%}.")
    prime = st.radio("Contractor", list(verdict.contractor), horizontal=True, key="pc_prime")
    d = pc[pc.contractor == prime]
    base = alt.Chart(d).encode(x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=0)))
    band = base.mark_area(opacity=0.10, color="#0b57d0").encode(y=alt.Y("lcl:Q", title="Share of lines with an issue", axis=alt.Axis(format="%")), y2="ucl:Q")
    ucl = base.mark_line(color=STATUS["critical"], strokeDash=[5, 3], interpolate="step-after").encode(y="ucl:Q")
    ctr = alt.Chart(pd.DataFrame({"c": [d.center.iloc[0]]})).mark_rule(color="#444746", strokeDash=[2, 2]).encode(y="c:Q")
    line = base.mark_line(color=SERIES[0], strokeWidth=2, point=alt.OverlayMarkDef(size=70, filled=True)).encode(
        y="p:Q", tooltip=["period", alt.Tooltip("p:Q", format=".1%", title="Issue rate"), alt.Tooltip("ucl:Q", format=".1%", title="Upper limit"),
                          alt.Tooltip("n:Q", title="Lines billed"), alt.Tooltip("x:Q", title="Lines with an issue")])
    out = base.mark_point(size=180, color=STATUS["critical"], shape="diamond", filled=True).encode(y="p:Q").transform_filter("datum.out_of_control")
    st.altair_chart((band + ucl + ctr + line + out).properties(height=300), width="stretch")
    howto("<b>How to read:</b> the dotted line is this contractor's average issue rate; the red dashed line is the upper control limit "
          "(average + 3 standard errors for that month's number of lines; months with fewer lines get wider limits). A point above the "
          "red line, 7 months in a row above average, or 5 rising months in a row is a <b>special cause</b>: something changed and is "
          "worth a meeting. Points inside the band are normal noise: reacting to them wastes everyone's time (Deming's tampering).")

    st.markdown("#### Segments: which review policy fits each package?")
    feat = PA.package_features(ctx.tables, ctx.gf, ctx.ef, ctx.pt)
    if len(feat) < 6:
        st.info("Needs at least 6 contract packages to segment.")
        return
    k = st.slider("Number of segments", 2, min(5, len(feat) - 1), 3, key="seg_k")
    seg, prof, sil = PA.segments(feat, k)
    seg["package"] = seg.project_id.map(lambda p: ctx.labels.get(p, p)) + " · " + seg.prime_id.map(lambda p: ctx.names.get(p, p))
    dom = [PA.POLICY[i][0] for i in range(3)]
    c1, c2 = st.columns([3, 2])
    sc = alt.Chart(seg).mark_circle(size=160, opacity=0.9, stroke="#ffffff", strokeWidth=2).encode(
        x=alt.X("pc1:Q", title="Component 1 (mostly first-pass rate and money at risk)"), y=alt.Y("pc2:Q", title="Component 2"),
        color=alt.Color("segment:N", scale=alt.Scale(domain=dom, range=[SERIES[2], SERIES[3], SERIES[1]]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["package", "segment", alt.Tooltip("first_pass_pct:Q", format=".0f"), alt.Tooltip("risk_pct:Q", format=".2f"),
                 alt.Tooltip("co_share_pct:Q", format=".1f"), alt.Tooltip("tm_share_pct:Q", format=".1f"), alt.Tooltip("spi:Q", format=".2f")])
    c1.altair_chart(sc.properties(height=320, title="Packages mapped to 2 dimensions (PCA)"), width="stretch")
    sl = sil.reset_index().rename(columns={"index": "k"})
    c2.altair_chart(alt.Chart(sl).mark_line(point=True, color=SERIES[0]).encode(
        x=alt.X("k:O", title="Number of segments"), y=alt.Y("silhouette:Q", title="Silhouette (higher = cleaner groups)")).properties(height=320,
        title="How many segments?"), width="stretch")
    st.dataframe(prof.reset_index().round(2), hide_index=True, width="stretch", column_config={
        "segment": "Segment", **{c: PA.SEG_FEATURES.get(c, c) for c in PA.SEG_FEATURES}, "bills": "Avg bills", "packages": "Packages"})
    pol = pd.DataFrame([{"segment": v[0], "review policy": v[1]} for v in PA.POLICY.values()])
    st.dataframe(seg[["package", "segment", "policy", "first_pass_pct", "risk_pct"]].sort_values("segment"), hide_index=True, width="stretch",
                 column_config={"package": "Package", "segment": "Segment", "policy": "Review policy", "first_pass_pct": st.column_config.NumberColumn("First-pass %", format="%.0f"),
                                "risk_pct": st.column_config.NumberColumn("At risk % of billed", format="%.2f")})
    howto("<b>How it works:</b> k-means groups packages on six standardized measures (first-pass rate, money at risk, schedule index, "
          "change-order share, T&M share, bill-to-bill jumps). Segments are ranked by risk and mapped to a policy: light touch saves "
          "reviewer hours on reliable packages; close watch puts them where errors are. With few packages, treat segments as a "
          "starting point and confirm with the control charts.")
    st.caption(" · ".join(f"**{p['segment']}**: {p['review policy']}" for p in pol.to_dict("records")))


# ---------------------------------------------------------------- Patterns
def patterns(ctx):
    back(ctx, "pt")
    hero("Patterns · what goes together, what comes first", "Which problems travel together?",
         "Association rules find issues that appear on the same bill. Lead-lag tests check whether one signal reliably comes before "
         "another. The learning curve asks whether repeated buildings get cheaper.")
    tabs = st.tabs(["Issue patterns", "Lead-lag tests", "Learning curve"])
    with tabs[0]:
        src = st.radio("Bills to mine", ["This program", "12-program training history (1,512 bills)"], horizontal=True, key="ar_src")
        b = PA.issue_baskets(ctx.gf, ctx.ef) if src == "This program" else PA.pooled_baskets()
        if b is None or not len(b):
            st.info("No training history found.")
        else:
            c = st.columns(2)
            ms = c[0].slider("Minimum support (share of bills with both)", 0.005, 0.05, 0.01, 0.005, key="ar_s")
            mc = c[1].slider("Minimum confidence", 0.05, 0.6, 0.15, 0.05, key="ar_c")
            rules, singles = PA.association_rules(b, ms, mc)
            if not len(rules):
                st.info("No pairs meet these thresholds. Lower them.")
            else:
                rules = rules.copy()
                rules["rule"] = rules.if_bill_has.map(lambda r: RULE_LABEL.get(r, r)) + "  →  " + rules.then_also.map(lambda r: RULE_LABEL.get(r, r))
                rules["significant"] = np.where(rules.p_value < 0.05, "p < 0.05", "not significant")
                top = rules.head(14)
                ch = alt.Chart(top).mark_bar(height=16, cornerRadiusEnd=3).encode(
                    y=alt.Y("rule:N", sort=list(top.rule), title=None, axis=alt.Axis(labelLimit=520)), x=alt.X("lift:Q", title="Lift (1 = independent)"),
                    color=alt.Color("significant:N", scale=alt.Scale(domain=["p < 0.05", "not significant"], range=[SERIES[0], "#80868b"]),
                                    legend=alt.Legend(orient="top", title=None)),
                    tooltip=["rule", alt.Tooltip("support:Q", format=".1%"), alt.Tooltip("confidence:Q", format=".0%"), alt.Tooltip("lift:Q", format=".2f"),
                             "bills_with_both", alt.Tooltip("p_value:Q", format=".3f", title="Fisher p")])
                one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color="#5e6368", strokeDash=[3, 3]).encode(x="x:Q")
                st.altair_chart((ch + one).properties(height=alt.Step(24)), width="stretch")
                t = rules.iloc[0]
                howto(f"<b>How to read:</b> <i>support</i> = share of bills with both issues; <i>confidence</i> = when the first appears, how "
                      f"often the second does too; <i>lift</i> = how many times more often than chance. <b>Top rule:</b> bills with "
                      f"{RULE_LABEL.get(t.if_bill_has, t.if_bill_has)} also have {RULE_LABEL.get(t.then_also, t.then_also)} {t.confidence:.0%} of "
                      f"the time, {t.lift:.1f}× chance ({int(t.bills_with_both)} bills, Fisher p = {t.p_value:.3f}). <b>Use it:</b> when a "
                      "reviewer finds the first issue, check for the second before releasing the bill. Small counts: treat grey bars as hints.")
    with tabs[1]:
        st.markdown("Granger test: do past values of one series improve the forecast of another, beyond that series' own past? "
                    "Pooled across the 12 synthetic training programs (contractor × month).")
        p = PA.granger_panel()
        c = st.columns(2)
        L = c[0].select_slider("Lags (months)", [1, 2], value=1, key="gr_L")
        fe = c[1].toggle("Control for month (time fixed effects)", True, key="gr_fe",
                         help="Without it, two series that both rise mid-project look like one leads the other.")
        pairs = [("co_approved_m", "error_rate"), ("error_rate", "co_approved_m"), ("co_approved_m", "overbilled_k"),
                 ("overbilled_k", "co_approved_m"), ("jump", "error_rate"), ("error_rate", "jump")]
        res = pd.DataFrame([PA.granger(p, x, y, L, fe) for x, y in pairs])
        alpha_adj = 0.05 / len(pairs)
        res["test"] = res.cause.map(PA.GRANGER_VARS) + "  →  " + res.effect.map(PA.GRANGER_VARS)
        res["verdict"] = np.where(res.p_value < alpha_adj, "Leads (Bonferroni-adjusted)", np.where(res.p_value < 0.05, "Weak (p < 0.05 only)", "No evidence"))
        ch = alt.Chart(res).mark_bar(height=18, cornerRadiusEnd=3).encode(
            y=alt.Y("test:N", sort=list(res.test), title=None, axis=alt.Axis(labelLimit=420)), x=alt.X("F:Q", title="F statistic"),
            color=alt.Color("verdict:N", scale=alt.Scale(domain=["Leads (Bonferroni-adjusted)", "Weak (p < 0.05 only)", "No evidence"],
                                                        range=[SERIES[2], SERIES[3], "#80868b"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["test", alt.Tooltip("F:Q", format=".2f"), alt.Tooltip("p_value:Q", format=".4f"), "n", alt.Tooltip("sum_of_cause_coefs:Q", format="+.3f", title="Effect")])
        st.altair_chart(ch.properties(height=alt.Step(30)), width="stretch")
        lead = res[res.verdict.str.startswith("Leads")]
        howto("<b>How to read:</b> a long green bar means the first series helps predict the second a month later. Six tests are run, "
              f"so the bar for 'leads' is p < {alpha_adj:.3f} (Bonferroni), not 0.05. <b>Result:</b> "
              + ("; ".join(f"{t}" for t in lead.test) + " passes." if len(lead) else "nothing passes the adjusted bar.")
              + " Toggle the time control off and watch weak 'leads' appear: they came from the project life-cycle, not from one series "
              "driving the other. Granger 'causality' means predictive lead, not proof of cause.")
        synthetic_note("Training programs are synthetic; their error drivers are documented in the README (a steep billing month raises the "
                       "chance of an error, which is why billing jumps lead errors here).")
    with tabs[2]:
        lc, hist = PR.learning_curve()
        c = lc["cost_per_mw"]
        k = st.columns(3)
        k[0].metric("Learning rate (cost per MW)", f"{c['rate']:.1%}", f"95% CI {c['rate_lo']:.1%} to {c['rate_hi']:.1%}", delta_color="off", delta_arrow="off",
                    help="Each doubling of buildings built multiplies cost per MW by this. 93% = 7% cheaper per doubling.")
        k[1].metric("Learning rate (build months)", f"{lc['build_months']['rate']:.1%}")
        k[2].metric("Fit (R², log-log)", f"{c['r2']:.2f}")
        grid = pd.DataFrame({"sequence": np.arange(1, 25)})
        grid["fit"] = PR.predict_curve(lc, grid.sequence) / 1e6
        pts = alt.Chart(hist.assign(c=hist.cost_per_mw / 1e6)).mark_circle(size=90, color=SERIES[0]).encode(
            x=alt.X("sequence:Q", title="Building number (cumulative)"), y=alt.Y("c:Q", title="Cost per MW ($M)", scale=alt.Scale(zero=False)),
            tooltip=["building", "year", "capacity_mw", alt.Tooltip("c:Q", format=".2f", title="$M/MW")])
        ln = alt.Chart(grid).mark_line(color=SERIES[1], strokeWidth=2).encode(x="sequence:Q", y="fit:Q")
        st.altair_chart((ln + pts).properties(height=300, title="Past buildings and the fitted curve (cost = a × n^b), extended to building 24"), width="stretch")
        n15, n20 = PR.predict_curve(lc, [15, 20])
        howto(f"<b>How it works:</b> regress log(cost per MW) on log(building number); the slope b gives the learning rate 2^b. "
              f"<b>Use it:</b> building 15 should cost about ${n15 / 1e6:.2f}M per MW and building 20 about ${n20 / 1e6:.2f}M. A bid well above "
              "the curve needs a reason (new design, new site, new contractor).")
        prog = PR.program_learning(ctx.tables, ctx.controls)
        if isinstance(prog, tuple):
            d, s = prog
            st.markdown(f"**This program's {s['n']} buildings (contract value per unit of scope, in start order):** learning rate "
                        f"{s['rate']:.1%}, p = {s['p']:.2f}. " + ("No learning visible: each building is priced independently. Ask the "
                                                                  "contractors to price repetition into the next buildings." if s["p"] > 0.1 else
                                                                  "A significant learning effect."))
        synthetic_note("Past-building history is synthetic (true learning rate 93%).")


# ---------------------------------------------------------------- Cash & process
def process(ctx):
    back(ctx, "pr")
    hero("Cash & process · flow of bills and money", "Where is the approval process slow, and how much cash do we need?",
         "Cycle time and Little's Law find the bottleneck in invoice approval. A seasonal model forecasts the capital spend the "
         "treasury needs to fund next quarter, and the program's plan gives a bottom-up check.")
    tabs = st.tabs(["Invoice cycle time", "Seasonal cash forecast"])
    with tabs[0]:
        log = ctx.extras.get("approval_log")
        if log is None or not len(log):
            st.info("Add approval_log.csv (invoice_id, stage, start, end) to see cycle times.")
        else:
            log = log[log.invoice_id.isin(ctx.gf.invoice_id)]
            ct = PR.cycle_time(log, ctx.gf)
            k = st.columns(5)
            k[0].metric("Average cycle time", f"{ct['cycle_days_mean']:.1f} days", f"P90 {ct['cycle_days_p90']:.0f} days", delta_color="off", delta_arrow="off")
            k[1].metric("Throughput", f"{ct['throughput_per_month']:.1f} bills / month")
            k[2].metric("Bills in process now", ct["wip_now"])
            k[3].metric("Little's Law check", f"{ct['little_cycle_days']:.1f} days", help="Average work-in-process ÷ throughput. Should be close to the measured cycle time.")
            k[4].metric("Bottleneck", ct["bottleneck"])
            s = ct["stages"]
            c1, c2 = st.columns([3, 2])
            bars = alt.Chart(s).mark_bar(height=20, cornerRadiusEnd=4).encode(
                y=alt.Y("stage:N", sort=list(s.stage), title=None), x=alt.X("mean_days:Q", title="Average days in stage"),
                color=alt.condition(alt.datum.stage == ct["bottleneck"], alt.value(SERIES[1]), alt.value(SERIES[0])),
                tooltip=["stage", alt.Tooltip("mean_days:Q", format=".1f"), alt.Tooltip("p90_days:Q", format=".1f", title="P90 days"), "bills",
                         alt.Tooltip("share_of_time_pct:Q", format=".0f", title="% of all waiting time"), "in_progress_now"])
            c1.altair_chart(bars.properties(height=alt.Step(34), title="Time in each approval stage"), width="stretch")
            pb = pd.DataFrame({"days": ct["per_bill"].values})
            c2.altair_chart(alt.Chart(pb).mark_bar(color=SERIES[2]).encode(x=alt.X("days:Q", bin=alt.Bin(maxbins=20), title="Days from submission to payment"),
                                                                             y=alt.Y("count():Q", title="Bills")).properties(height=230, title="Cycle time per bill"), width="stretch")
            bn = s.set_index("stage").loc[ct["bottleneck"]]
            howto(f"<b>How to read:</b> {ct['bottleneck']} holds {bn.share_of_time_pct:.0f}% of all waiting time ({bn.mean_days:.1f} days on "
                  f"average across {int(bn.bills)} bills). <b>Little's Law</b> (work-in-process = throughput × cycle time) says the queue "
                  f"averages {ct['avg_wip']:.1f} bills at {ct['throughput_per_month']:.1f} bills a month, so a bill waits {ct['little_cycle_days']:.0f} days. "
                  "<b>Decisions:</b> cut field-verification time with site photos and receipts entered on the day; send disputes to a "
                  "weekly commercial meeting with a 10-day target. Late payment costs contractor goodwill and can trigger prompt-payment interest.")
            synthetic_note("Stage timestamps are synthetic, generated to match each bill's real submitted and paid dates.")
    with tabs[1]:
        with st.spinner("Fitting SARIMA and Holt-Winters..."):
            cf = _cash()
        acc = cf["accuracy"]
        best = cf["best"]
        fc = cf["forecast"]
        h = cf["history"].copy()
        h["date"] = pd.PeriodIndex(h.period, freq="M").to_timestamp()
        f = fc[fc.method == best].copy()
        f["date"] = pd.PeriodIndex(f.period, freq="M").to_timestamp()
        k = st.columns(4)
        nq = f.head(3)
        k[0].metric("Next 3 months (best model)", money(nq.forecast.sum()), f"80% range {money(nq.lo80.sum())} – {money(nq.hi80.sum())}", delta_color="off", delta_arrow="off")
        k[1].metric("Best model (backtest MAPE)", best.split("(")[0].strip(), f"{acc.set_index('method').loc[best, 'MAPE (%)']:.1f}% average error", delta_color="off", delta_arrow="off")
        k[2].metric("Peak month ahead", f"{f.loc[f.forecast.idxmax(), 'period']}", money(f.forecast.max()), delta_color="off", delta_arrow="off")
        k[3].metric("Seasonal swing", f"{100 * (cf['season'].max() - cf['season'].min()):.0f} pts", f"high {cf['season'].idxmax()}, low {cf['season'].idxmin()} (month)",
                    delta_color="off", delta_arrow="off")
        hist_l = alt.Chart(h[h.date >= "2022-01-01"]).mark_line(color=SERIES[0], strokeWidth=2).encode(
            x=alt.X("date:T", title=None), y=alt.Y("capital_spend:Q", title="Monthly capital spend ($)", axis=alt.Axis(format="$~s")),
            tooltip=["period", alt.Tooltip("capital_spend:Q", format="$,.0f")])
        band = alt.Chart(f).mark_area(opacity=0.2, color=SERIES[1]).encode(x="date:T", y="lo80:Q", y2="hi80:Q")
        fl = alt.Chart(f).mark_line(color=SERIES[1], strokeWidth=2, strokeDash=[5, 3], point=True).encode(
            x="date:T", y="forecast:Q", tooltip=["period", alt.Tooltip("forecast:Q", format="$,.0f"), alt.Tooltip("lo80:Q", format="$,.0f"), alt.Tooltip("hi80:Q", format="$,.0f")])
        st.altair_chart((hist_l + band + fl).properties(height=300, title=f"Owner's capital spend and the {best} forecast (80% band)"), width="stretch")
        c1, c2 = st.columns([3, 2])
        c1.dataframe(acc.round(2), hide_index=True, width="stretch")
        se = cf["season"].reset_index()
        se.columns = ["month", "index"]
        c2.altair_chart(alt.Chart(se).mark_bar(color=SERIES[2], cornerRadiusEnd=3).encode(
            x=alt.X("month:O", title="Month"), y=alt.Y("index:Q", title="Seasonal index (1 = average)", scale=alt.Scale(zero=False))).properties(height=200), width="stretch")
        howto("<b>How it works:</b> three models are backtested at 3-month horizons over the last two years (rolling origin): "
              "SARIMA on log spend, Holt-Winters with multiplicative seasonality, and a seasonal-naive benchmark (same month last year). "
              f"The model with the lowest MAPE ({best}) makes the forecast. <b>Use it:</b> treasury funds the P80 (top of band), not the "
              "mean; December's year-end push and the January-February dip are the months to plan for.")
        cal = PR.program_cash_calendar(ctx.controls, ctx.pt, ctx.tables, ctx.end_period)
        if len(cal):
            cal = cal[cal.project_id.isin(ctx.pt.project_id)]
            cc = cal.groupby("pay_period").cash_out.sum().reset_index()
            st.markdown(f"**Bottom-up check, this program:** the remaining baseline plan bills {money(cal.billing.sum())}, paid "
                        f"{money(cc.cash_out.sum())} net of retainage over {len(cc)} months (payment one month after billing).")
            st.altair_chart(alt.Chart(cc).mark_bar(color=SERIES[3], cornerRadiusEnd=3).encode(
                x=alt.X("pay_period:O", title=None), y=alt.Y("cash_out:Q", title="Cash out ($)", axis=alt.Axis(format="$~s")),
                tooltip=["pay_period", alt.Tooltip("cash_out:Q", format="$,.0f")]).properties(height=200), width="stretch")
        synthetic_note("The 8-year capital-spend history is synthetic (trend + seasonality + noise). Load the owner's real monthly AP or "
                       "capex history to use it.")


@st.cache_data(show_spinner=False)
def _cash():
    return PR.cash_forecast()


# ---------------------------------------------------------------- Sustainability
def carbon(ctx):
    back(ctx, "cb")
    hero("Sustainability · embodied carbon", "What is the carbon cost of the build?",
         "Contract quantities times material carbon factors give each project's embodied carbon (cradle to gate). Equipment uses a "
         "spend-based factor. The hotspots show where low-carbon concrete or recycled steel would matter most.")
    cb = PR.carbon(ctx.tables)
    bp = cb["by_project"]
    bp = bp[bp.project_id.isin(ctx.pt.project_id)]
    if not len(bp):
        st.info("None of this program's items have carbon factors. Add rows to data/market/carbon_factors.csv.")
        return
    bp["project"] = bp.project_id.map(lambda p: ctx.labels.get(p, p))
    k = st.columns(4)
    k[0].metric("Embodied carbon", f"{bp.tco2e.sum():,.0f} tCO₂e")
    k[1].metric("Intensity", f"{bp.tco2e.sum() / (bp.contract_value.sum() / 1e6):,.0f} t per $1M")
    k[2].metric("Contract value covered", f"{100 * cb['coverage']:.0f}%", help="Share of contract value with a carbon factor (the rest is labour-only or unmapped).")
    top = cb["by_material"].iloc[0]
    k[3].metric("Largest source", top.material, f"{top.tco2e:,.0f} t", delta_color="off", delta_arrow="off")
    c1, c2 = st.columns(2)
    c1.altair_chart(alt.Chart(bp).mark_bar(height=20, cornerRadiusEnd=4, color=SERIES[2]).encode(
        y=alt.Y("project:N", sort="-x", title=None), x=alt.X("tco2e:Q", title="tCO₂e"),
        tooltip=["project", alt.Tooltip("tco2e:Q", format=",.0f"), alt.Tooltip("t_per_musd:Q", format=",.0f", title="t per $1M")]).properties(
        height=alt.Step(30), title="By project"), width="stretch")
    bm = cb["by_material"]
    c2.altair_chart(alt.Chart(bm).mark_bar(height=16, cornerRadiusEnd=4, color=SERIES[0]).encode(
        y=alt.Y("material:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("tco2e:Q", title="tCO₂e"),
        tooltip=["material", alt.Tooltip("tco2e:Q", format=",.0f")]).properties(height=alt.Step(24), title="By material (hotspots)"), width="stretch")
    st.markdown("#### What-if: low-carbon choices")
    c = st.columns(3)
    conc = c[0].slider("Concrete: cement replacement cuts factor by", 0, 50, 30, 5, format="%d%%", key="cb_c")
    stl = c[1].slider("Steel: recycled-content (EAF) cuts factor by", 0, 60, 40, 5, format="%d%%", key="cb_s")
    prem = c[2].slider("Cost premium for low-carbon materials", 0.0, 5.0, 1.0, 0.5, format="%.1f%%", key="cb_p")
    lines = cb["lines"]
    lines = lines[lines.project_id.isin(ctx.pt.project_id) & (lines.method != "not covered")]
    is_c = lines.material.str.contains("Concrete|concrete", na=False)
    is_s = lines.material.str.contains("steel|Steel", na=False)
    saved = (lines.tco2e[is_c] * conc / 100).sum() + (lines.tco2e[is_s] * stl / 100).sum()
    cost = (lines.scheduled_value[is_c | is_s].fillna(0) * prem / 100).sum()
    k = st.columns(3)
    k[0].metric("Carbon saved", f"{saved:,.0f} tCO₂e", f"{100 * saved / bp.tco2e.sum():.0f}% of the total", delta_color="normal")
    k[1].metric("Extra cost", money(cost))
    k[2].metric("Cost per tonne avoided", money(cost / saved) if saved else "–", help="Compare with an internal carbon price (often $50-$200/t).")
    howto("<b>How to read:</b> reinforcing steel and structural concrete dominate the data center foundations, so specifying "
          "cement-replacement concrete and recycled-content rebar moves the total most. The cost per tonne avoided tells you whether "
          "the switch beats buying offsets or the company's internal carbon price.")
    with st.expander("Carbon factors used"):
        st.dataframe(pd.read_csv(PR.MARKET / "carbon_factors.csv"), hide_index=True, width="stretch")
        st.caption("Keyword factors for canal work: concrete lining 0.11 kgCO₂e/kg at 2,400 kg/m³, LDPE film 2.0, brick 0.24, galvanized pipe 2.76.")
    synthetic_note("Factors are approximate generic cradle-to-gate values in the style of the ICE database, with assumed weights per "
                   "unit. Use product EPDs for any reported figure.")
