"""ML Lab (bill risk model + reviewer plan) and Forecast & risk (Monte Carlo, tornado, backtest) pages."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import forecast_risk as F
import risk_model as R
from insights import money
from ui.theme import SERIES, STATUS

SIMPLICITY = ["Logistic regression", "CART (depth 4)", "Random forest", "Gradient boosting", "Stacked ensemble"]


def _howto(text):
    st.markdown(f'<div class="howto">{text}</div>', unsafe_allow_html=True)


def _back(ctx):
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        st.button("← Insights", key=f"bk_ins_{st.session_state.get('_pg', '')}", on_click=ctx.go, args=("insights", None, None), type="tertiary")
    with b2:
        st.button("⌂ Mission Control", key=f"bk_home_{st.session_state.get('_pg', '')}", on_click=ctx.go, args=("home", None, None), type="tertiary")


@st.cache_resource(show_spinner="Training risk models on 12 synthetic programs...")
def _trained(mtime):
    df = pd.read_csv(R.TRAIN_FILE)
    tr, ho = df[df.split == "train"], df[df.split == "holdout"]
    models, res, rocs, scores = R.train_and_evaluate(tr, ho)
    sel_file = R.TRAIN_FILE.parent / "model_selection.csv"
    sel = pd.read_csv(sel_file) if sel_file.exists() else pd.DataFrame({"model": list(models), "cv_auc": np.nan})
    res = res.merge(sel, on="model", how="left")
    best_cv = sel.cv_auc.max()
    chosen = next(m for m in SIMPLICITY if m in set(sel[sel.cv_auc >= best_cv - 0.005].model))
    ratio = (tr[tr.has_error == 1].overbilled / tr[tr.has_error == 1].gross_amount).median()
    return {"models": models, "res": res, "rocs": rocs, "scores": scores, "holdout": ho, "train": tr,
            "chosen": chosen, "overbill_ratio": float(ratio)}


# ---------------------------------------------------------------- ML Lab
def ml_lab(ctx):
    st.session_state["_pg"] = "ml"
    _back(ctx)
    st.markdown('<div class="eyebrow">ML Lab · bill risk</div><div class="hero-title">Which bills should we check first?</div>'
                '<div class="hero-sub">A model scores every incoming bill for the chance it contains an error, using only what is '
                'known before checking it. The rules engine still decides payment; the score decides the order of review.</div>',
                unsafe_allow_html=True)
    st.warning("Method demo. The models train and test on synthetic programs from the same generator, so they may be learning "
               "how the generator plants errors, not how real contractors bill. The scores below show the method works, not that it "
               "would rank real bills well. Real use needs reviewer-confirmed labels, a test on later billing months, and a "
               "measure of confirmed dollars found per reviewer hour.", icon="⚗️")
    if not R.TRAIN_FILE.exists():
        st.error("Training data not found (data/training/bill_features.csv). Run `python src/build_training.py`.")
        return
    T = _trained(R.TRAIN_FILE.stat().st_mtime)
    res, ho, chosen = T["res"], T["holdout"], T["chosen"]
    row = res.set_index("model").loc[chosen]
    k = st.columns(4)
    k[0].metric("Model in use", chosen, help="Chosen by 5-fold cross-validation on the training programs (simplest model "
                "within 0.005 AUC of the best). The hold-out program played no part in the choice.")
    k[1].metric("Hold-out AUC", f"{row.auc:.2f}", help="Area under the ROC curve on a program the model never saw. "
                "0.5 = coin flip, 1.0 = perfect. 0.70-0.80 is a useful real-world ranking model.")
    k[2].metric("Errors caught in top 20%", f"{row.errors_caught_top20:.0%}", f"{row.errors_caught_top20 / 0.2:.1f}× random",
                delta_color="off", delta_arrow="off", help="Review the riskiest 20% of bills: share of all error bills you reach.")
    k[3].metric("Overbilling caught in top 20%", f"{row.money_caught_top20:.0%}", f"{row.money_caught_top20 / 0.2:.1f}× random",
                delta_color="off", delta_arrow="off", help="Same, measured in overbilled dollars.")
    _howto(f"<b>What this means:</b> on {len(ho)} bills it had never seen, the {chosen.lower()} put the riskiest bills first well "
           f"enough that checking the top 20% reaches {row.errors_caught_top20:.0%} of the bills with errors, about "
           f"{row.errors_caught_top20 / 0.2:.1f} times better than checking at random. Errors are partly random by nature, so no "
           "model reaches 100%; the gain is in where reviewers start.")

    st.markdown("### Model comparison (hold-out program)")
    show = res.copy()
    st.dataframe(show[["model", "cv_auc", "auc", "errors_caught_top20", "money_caught_top20"]], width="stretch", hide_index=True,
                 column_config={"model": "Model", "cv_auc": st.column_config.NumberColumn("CV AUC (train)", format="%.3f"),
                                "auc": st.column_config.NumberColumn("Hold-out AUC", format="%.3f"),
                                "errors_caught_top20": st.column_config.NumberColumn("Errors in top 20%", format="percent"),
                                "money_caught_top20": st.column_config.NumberColumn("$ in top 20%", format="percent")})
    c1, c2 = st.columns(2)
    rocs = T["rocs"]
    dom = list(res.model)
    roc = alt.Chart(rocs).mark_line(strokeWidth=2).encode(
        x=alt.X("fpr:Q", title="False alarm rate (clean bills flagged)", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("tpr:Q", title="Error bills caught", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=alt.Scale(domain=dom, range=SERIES[:len(dom)]), legend=alt.Legend(orient="bottom", title=None, columns=2)),
        tooltip=["model", alt.Tooltip("fpr:Q", format=".2f"), alt.Tooltip("tpr:Q", format=".2f")])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="#8792a6").encode(x="x:Q", y="y:Q")
    c1.altair_chart((roc + diag).properties(height=330, title="ROC curves"), width="stretch")
    g = R.gains(ho.assign(score=T["scores"][chosen]))
    gl = g.melt(id_vars="share_reviewed", value_vars=["share_errors", "share_money"], var_name="m", value_name="v")
    gl["m"] = gl.m.map({"share_errors": "Error bills caught", "share_money": "Overbilled $ caught"})
    rnd = pd.DataFrame({"share_reviewed": [0, 1], "v": [0, 1], "m": "Random order"})
    gd = ["Error bills caught", "Overbilled $ caught", "Random order"]
    gc = alt.Chart(pd.concat([gl, rnd])).mark_line(strokeWidth=2).encode(
        x=alt.X("share_reviewed:Q", title="Share of bills reviewed (riskiest first)", axis=alt.Axis(format="%")),
        y=alt.Y("v:Q", title="Share caught", axis=alt.Axis(format="%")),
        color=alt.Color("m:N", scale=alt.Scale(domain=gd, range=[SERIES[0], SERIES[1], "#8792a6"]), legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition("datum.m == 'Random order'", alt.value([4, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("share_reviewed:Q", format=".0%"), "m", alt.Tooltip("v:Q", format=".0%")])
    rule = alt.Chart(pd.DataFrame({"x": [0.2]})).mark_rule(color="#8792a6", strokeDash=[2, 3]).encode(x="x:Q")
    c2.altair_chart((gc + rule).properties(height=330, title=f"Gains chart: {chosen}"), width="stretch")
    _howto("<b>How to read:</b> ROC: the further a curve bows toward the top-left, the better the model separates bills with "
           "errors from clean ones; the dashed diagonal is guessing. Gains: follow the dotted line at 20% reviewed up to each curve "
           "to see how much you catch; the gap above the dashed random line is the value of the model.")

    st.markdown("### Threshold and confusion matrix")
    top = st.slider("Flag the riskiest share of bills for review", 0.05, 0.6, 0.3, 0.05, format="%.2f")
    p = T["scores"][chosen]
    thr = float(np.quantile(p, 1 - top))
    cm = R.confusion(ho.has_error.values, p, thr)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(cm, width="stretch")
    tp, fn, fp = cm.iloc[0, 0], cm.iloc[0, 1], cm.iloc[1, 0]
    c2.markdown(f'<div class="howto">Flagging the top <b>{top:.0%}</b> of bills: <b>{tp}</b> of {tp + fn} error bills are flagged '
                f'(sensitivity {tp / max(tp + fn, 1):.0%}); <b>{fp}</b> clean bills are flagged too (they cost review time, not '
                f'money); precision {tp / max(tp + fp, 1):.0%}. A lower share saves reviewer time but misses more; a higher share '
                'catches more but reviews more clean bills.</div>', unsafe_allow_html=True)

    st.markdown("### What drives the risk")
    coef = R.lr_coefficients(T["models"]["Logistic regression"])
    coef["direction"] = np.where(coef.coef >= 0, "Raises risk", "Lowers risk")
    coef["one"] = 1.0
    cb = alt.Chart(coef).mark_bar(height=14).encode(
        y=alt.Y("feature:N", sort=list(coef.feature), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("odds_ratio_per_sd:Q", title="Odds ratio per standard deviation (1 = no effect)", scale=alt.Scale(zero=False)),
        x2="one:Q",
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[SERIES[1], SERIES[0]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["feature", alt.Tooltip("odds_ratio_per_sd:Q", format=".2f")])
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color="#8792a6").encode(x="x:Q")
    st.altair_chart((cb + one).properties(height=alt.Step(24), title="Logistic regression: effect of each signal"), width="stretch")
    _howto("<b>How to read:</b> an odds ratio of 1.5 means a bill one standard deviation higher on that signal has 1.5× the odds "
           "of containing an error, holding the others fixed. These are associations in the training data, not proof of cause.")
    with st.expander("Decision tree rules (CART): a readable version for auditors"):
        st.code(R.tree_rules(T["models"]["CART (depth 4)"]), language="text")

    # ---------------- reviewer plan on current data
    st.markdown("### This month's review plan")
    if ctx.structure_of(ctx.pt.project_id.iloc[0]) != "datacenter":
        st.warning("The model was trained on data center bills. Scores for this program are shown for illustration only.")
    feats = R.bill_features(ctx.tables, ctx.gates)
    feats = feats[feats.invoice_id.isin(ctx.gf.invoice_id)]
    month = st.select_slider("Bills received in", options=sorted(feats.period.unique()), value=sorted(feats.period.unique())[-1])
    cur = feats[feats.period == month].copy()
    model = T["models"][chosen]
    cur["risk"] = model.predict_proba(cur[R.FEATURES].values)[:, 1]
    cur["expected_money"] = cur.risk * T["overbill_ratio"] * cur.gross_amount
    cur["why"] = R.reasons(T["models"]["Logistic regression"], cur)
    hours = st.slider("Reviewer hours available this month", 2.0, 60.0, 12.0, 1.0,
                      help="Review time per bill = 0.5 h + 0.08 h per line (assumption).")
    plan = R.review_plan(cur, hours)
    picked = plan[plan.review]
    rand_value = plan.expected_money.mean() * len(picked)
    k = st.columns(4)
    k[0].metric("Bills to review", f"{len(picked)} of {len(plan)}")
    k[1].metric("Hours used", f"{picked.review_hours.sum():.1f} of {hours:.0f}")
    k[2].metric("Expected overbilling covered", money(picked.expected_money.sum()),
                f"{picked.expected_money.sum() / max(plan.expected_money.sum(), 1):.0%} of total", delta_color="off", delta_arrow="off")
    k[3].metric("Rules later flagged", f"{int((picked.gate != 'PASS').sum())} of {len(picked)}",
                help="On demo data you can check the model: how many of the picked bills the rules engine actually stopped or flagged.")
    _howto(f"<b>How it works:</b> an optimization (0-1 knapsack, solved as a mixed-integer program) picks the set of bills that covers "
           f"the most expected overbilling within {hours:.0f} hours. Expected overbilling = risk × typical overbill size "
           f"({T['overbill_ratio']:.1%} of a bill with errors) × bill value. Reviewing the same number of bills at random would cover "
           f"about {money(rand_value)}.")
    view = plan.sort_values(["review", "risk"], ascending=[False, False])
    view["project"] = view.project_id.map(lambda x: ctx.labels.get(x, x)) if "project_id" in view else ""
    view["contractor"] = view.prime_id.map(lambda x: ctx.names.get(x, x))
    st.dataframe(view[["review", "invoice_id", "project", "contractor", "risk", "gross_amount", "expected_money", "review_hours", "why", "gate"]],
                 width="stretch", hide_index=True, column_config={
                     "review": st.column_config.CheckboxColumn("Review"), "invoice_id": "Bill",
                     "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="percent"),
                     "gross_amount": st.column_config.NumberColumn("Billed", format="$%,.0f"),
                     "expected_money": st.column_config.NumberColumn("Expected overbilling", format="$%,.0f"),
                     "review_hours": st.column_config.NumberColumn("Hours", format="%.1f"),
                     "why": "Top reasons", "gate": "Rules result"})
    st.caption("Honest note: the training programs are synthetic, generated with documented risk drivers (contractor habits, "
               "first bills, rushed months, pending change orders, repeat errors). The model shows the method; real accuracy needs "
               "real review history.")


# ---------------------------------------------------------------- Forecast & risk
@st.cache_data(show_spinner="Running 5,000 scenarios...")
def _mc(_tables, key, row_dict, prog_p, assumptions):
    row = pd.Series(row_dict)
    mc = F.monte_carlo(_tables, row, prog_p, n=5000, a=dict(assumptions))
    return mc, F.summarize(mc, row)


@st.cache_data(show_spinner="Running the sensitivity analysis...")
def _tornado(_tables, key, row_dict, prog_p, assumptions):
    return F.tornado(_tables, pd.Series(row_dict), prog_p, a=dict(assumptions))


def forecast(ctx):
    st.session_state["_pg"] = "fc"
    _back(ctx)
    st.markdown('<div class="eyebrow">Forecast & risk</div><div class="hero-title">What will it really cost, and when will it finish?</div>'
                '<div class="hero-sub">5,000 simulated futures per project turn single-number forecasts into ranges, show what '
                'drives them, and test how accurate the forecasting method has been.</div>', unsafe_allow_html=True)
    pt = ctx.pt
    ids = list(pt.project_id)
    cur = st.session_state.get("project") if st.session_state.get("project") in ids else ids[0]
    pick = st.selectbox("Project", ids, index=ids.index(cur), format_func=lambda p: ctx.labels.get(p, p))
    st.session_state["project"] = pick
    r = pt.set_index("project_id").loc[pick].copy()
    r["project_id"] = pick
    pp = ctx.prog[ctx.prog.project_id == pick]
    with st.expander("Assumptions (edit to test your own view)"):
        a = dict(F.DEFAULTS)
        c = st.columns(4)
        a["p_approve"] = c[0].slider("Chance a pending change is approved", 0.0, 1.0, a["p_approve"], 0.05)
        a["negotiate_lo"] = c[1].slider("Lowest settlement (% of ask)", 0.4, 1.0, a["negotiate_lo"], 0.05)
        a["growth_max"] = c[2].slider("Unforeseen new changes, max (% of remaining)", 0.0, 10.0, a["growth_max"], 0.5)
        a["growth_mode"] = min(a["growth_mode"], a["growth_max"])
        a["esc_mean"] = c[3].slider("Material price change, mean (%)", -5.0, 10.0, a["esc_mean"], 0.5)
        c = st.columns(4)
        a["esc_sd"] = c[0].slider("Material price change, spread (sd %)", 0.0, 8.0, a["esc_sd"], 0.5)
        a["material_share"] = c[1].slider("Materials share of remaining work (%)", 10.0, 70.0, a["material_share"], 5.0)
        a["pace_cv_floor"] = c[2].slider("Minimum schedule uncertainty (cv)", 0.05, 0.6, a["pace_cv_floor"], 0.05)
        st.caption("Pending change orders: each approved with the chance above, settling uniformly between the lowest settlement and "
                   "100%. New changes: triangular from 0 to the max. Prices: normal. Remaining duration: lognormal around the "
                   "earned-schedule forecast, spread from this project's month-to-month progress variability.")
    keep = ["project_id", "committed", "original_contract", "approved_co", "budget", "verified", "verified_pct",
            "planned_finish", "forecast_finish", "data_date", "ld_rate", "ld_cap"]
    row_dict = {k_: r.get(k_) for k_ in keep}
    key = (pick, ctx.end_period, tuple(sorted(a.items())))
    mc, s = _mc(ctx.tables, key, row_dict, pp, tuple(sorted(a.items())))
    k = st.columns(4)
    k[0].metric("Cost P50 (median)", money(s["cost_p50"]), help="Half the simulated futures cost less than this.")
    k[1].metric("Cost P80", money(s["cost_p80"]), help="80% of futures cost less: a common level to hold contingency to.")
    if "p_within_budget" in s:
        k[2].metric("Chance within budget", f"{s['p_within_budget']:.0%}")
        need = s["contingency_for_p80"]
        k[3].metric("Budget vs P80", money(abs(need)), "short at P80" if need > 0 else "headroom at P80",
                    delta_color="inverse" if need > 0 else "normal")
    if "finish_p50" in s:
        k = st.columns(4)
        k[0].metric("Finish P50", f"{s['finish_p50']:%d %b %Y}")
        k[1].metric("Finish P80", f"{s['finish_p80']:%d %b %Y}")
        if "p_on_time" in s:
            k[2].metric("Chance on time", f"{s['p_on_time']:.0%}", help="Finishing on or before the contract completion date.")
        if "ld_p80" in s:
            k[3].metric("Penalty exposure P80", money(s["ld_p80"]))

    c1, c2 = st.columns(2)
    hist = pd.DataFrame({"cost": mc["cost"] / 1e6})
    marks = [{"x": s["cost_p50"] / 1e6, "t": "P50"}, {"x": s["cost_p80"] / 1e6, "t": "P80"}]
    if pd.notna(r.get("budget")):
        marks.append({"x": r["budget"] / 1e6, "t": "Budget"})
    mk = pd.DataFrame(marks)
    h = alt.Chart(hist).mark_bar(color=SERIES[0], opacity=0.85).encode(
        x=alt.X("cost:Q", bin=alt.Bin(maxbins=45), title="Final cost ($M)"), y=alt.Y("count():Q", title="Scenarios"))
    rl = alt.Chart(mk).mark_rule(strokeDash=[4, 3]).encode(x="x:Q", color=alt.Color("t:N", scale=alt.Scale(
        domain=["P50", "P80", "Budget"], range=["#e8eef8", SERIES[3], STATUS["critical"]]), legend=alt.Legend(orient="top", title=None)))
    c1.altair_chart((h + rl).properties(height=280, title="Final cost: 5,000 scenarios"), width="stretch")
    if "finish" in mc:
        wk = pd.to_datetime(mc["finish"]).to_period("W").start_time
        fh = pd.Series(wk).value_counts().rename_axis("finish").reset_index(name="n")
        fm = [{"x": s["finish_p50"], "t": "P50"}, {"x": s["finish_p80"], "t": "P80"}]
        if pd.notna(r.get("planned_finish")):
            fm.append({"x": r["planned_finish"], "t": "Contract finish"})
        fmk = pd.DataFrame(fm)
        fhc = alt.Chart(fh).mark_bar(color=SERIES[2], opacity=0.85, width=9).encode(
            x=alt.X("finish:T", title="Finish date (week)", axis=alt.Axis(format="%d %b %y")), y=alt.Y("n:Q", title="Scenarios"),
            tooltip=[alt.Tooltip("finish:T", format="week of %d %b %Y"), alt.Tooltip("n:Q", title="Scenarios")])
        frl = alt.Chart(fmk).mark_rule(strokeDash=[4, 3]).encode(x="x:T", color=alt.Color("t:N", scale=alt.Scale(
            domain=["P50", "P80", "Contract finish"], range=["#e8eef8", SERIES[3], STATUS["critical"]]), legend=alt.Legend(orient="top", title=None)))
        c2.altair_chart((fhc + frl).properties(height=280, title="Finish date: 5,000 scenarios"), width="stretch")
    _howto(f"<b>How to read:</b> each bar counts simulated futures. The wider the spread, the less certain the outcome. "
           f"<b>Now:</b> cost lands between {money(s['cost_p10'])} and {money(s['cost_p90'])} in 80% of futures"
           + (f"; finish between {s['finish_p10']:%d %b} and {s['finish_p90']:%d %b %Y}" if "finish_p10" in s else "")
           + (f". The budget covers {s['p_within_budget']:.0%} of futures." if "p_within_budget" in s else "."))

    st.markdown("### What drives the forecast (tornado)")
    td, b_cost = _tornado(ctx.tables, key, row_dict, pp, tuple(sorted(a.items())))
    tc = td[td.measure == "cost"].sort_values("swing", ascending=False)
    tl = pd.concat([tc.assign(side="Low value", v=tc.cost_low), tc.assign(side="High value", v=tc.cost_high)])
    tl["lab"] = tl.v.map(lambda v: ("+" if v >= 0 else "−") + money(abs(v)))
    tor = alt.Chart(tl).mark_bar(height=18).encode(
        y=alt.Y("input:N", sort=list(tc.input), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("v:Q", title="Change in median final cost ($)", axis=alt.Axis(format="$~s")),
        color=alt.Color("side:N", scale=alt.Scale(domain=["Low value", "High value"], range=[SERIES[0], SERIES[1]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["input", "side", alt.Tooltip("v:Q", format="$,.0f")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8792a6").encode(x="x:Q")
    st.altair_chart((tor + zero).properties(height=alt.Step(34)), width="stretch")
    top = tc.iloc[0]
    fin = td[td.measure == "finish"]
    _howto(f"<b>How to read:</b> each bar shows how far the median cost moves when one input goes from its low to its high value, "
           f"the others held at their base. Longest bar = the input to manage first. <b>Now:</b> \"{top.input}\" moves cost by "
           f"{money(top.swing)}."
           + (f" Schedule uncertainty moves the P80 finish by about {int(fin.swing.iloc[0])} days." if len(fin) else ""))

    st.markdown("### All projects")
    rows = []
    for _, rr in pt.iterrows():
        rd = {k_: rr.get(k_) for k_ in keep}
        _, ss = _mc(ctx.tables, (rr.project_id, ctx.end_period, tuple(sorted(a.items()))), rd,
                    ctx.prog[ctx.prog.project_id == rr.project_id], tuple(sorted(a.items())))
        rows.append({"project": rr["name"], "budget": rr.get("budget"), "p50": ss["cost_p50"], "p80": ss["cost_p80"],
                     "within": ss.get("p_within_budget"), "fin80": ss.get("finish_p80"), "on_time": ss.get("p_on_time")})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, column_config={
        "project": "Project", "budget": st.column_config.NumberColumn("Budget", format="$%,.0f"),
        "p50": st.column_config.NumberColumn("Cost P50", format="$%,.0f"), "p80": st.column_config.NumberColumn("Cost P80", format="$%,.0f"),
        "within": st.column_config.NumberColumn("Chance within budget", format="percent"),
        "fin80": st.column_config.DateColumn("Finish P80"), "on_time": st.column_config.NumberColumn("Chance on time", format="percent")})

    st.markdown("### How accurate are the forecasts? (backtest)")
    bt, summ = F.backtest(ctx.prog[ctx.prog.project_id.isin(pt.project_id)])
    if not len(summ):
        st.caption("Not enough months of history to backtest.")
        return
    summ["horizon"] = summ.h.map(lambda x: f"{x} month{'s' if x > 1 else ''} ahead")
    bc = alt.Chart(summ).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("horizon:N", title=None, axis=alt.Axis(labelAngle=0)), xOffset="method:N",
        y=alt.Y("MAE (pts):Q", title="Average error (progress points)"),
        color=alt.Color("method:N", scale=alt.Scale(domain=["Plan-adjusted", "Recent pace"], range=[SERIES[2], SERIES[1]]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["method", "horizon", alt.Tooltip("MAE (pts):Q", format=".2f"), alt.Tooltip("MAPE (%):Q", format=".1f"),
                 alt.Tooltip("Bias (pts):Q", format="+.2f"), "tests"])
    c1, c2 = st.columns([2, 3])
    c1.altair_chart(bc.properties(height=260), width="stretch")
    c2.dataframe(summ[["method", "horizon", "tests", "MAE (pts)", "RMSE (pts)", "MAPE (%)", "Bias (pts)"]].round(2),
                 width="stretch", hide_index=True)
    best = summ[summ.h == 1].sort_values("MAE (pts)").iloc[0]
    worst = summ[summ.h == 1].sort_values("MAE (pts)").iloc[-1]
    _howto(f"<b>How it works:</b> at each past month we pretend the future is unknown, forecast progress one and two months ahead, "
           f"then compare with what actually happened (a pseudo-out-of-sample test). <b>Result:</b> the {best.method.lower()} method "
           f"was off by {best['MAE (pts)']:.1f} points a month ahead vs {worst['MAE (pts)']:.1f} for {worst.method.lower()}, so the "
           "dashboard's finish forecasts use earned schedule (a plan-adjusted method). Caveat: in the data center demo the plan was "
           "built from actual progress, which favors plan-based methods; real plans are noisier.")
