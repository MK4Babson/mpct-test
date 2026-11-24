"""Cost intelligence (beta/alpha, should-cost, market benchmark, Cost-at-Risk) and Change orders (decision tree + analyst)."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ai
import change_orders as CO
import cost_intel as CI
from insights import money
from ui.theme import SERIES, STATUS

DOLLAR = st.column_config.NumberColumn(format="$%,.0f")


def howto(text):
    st.markdown(f'<div class="howto">{text}</div>', unsafe_allow_html=True)


def hero(eyebrow, title, sub):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><div class="hero-sub">{sub}</div>',
                unsafe_allow_html=True)


def back(ctx, key):
    b1, b2, _ = st.columns([1, 1, 6])
    b1.button("← Insights", key=f"bk1_{key}", on_click=ctx.go, args=("insights", None, None), type="tertiary")
    b2.button("⌂ Mission Control", key=f"bk2_{key}", on_click=ctx.go, args=("home", None, None), type="tertiary")


def synthetic_note(text):
    st.caption(f"🧪 {text}")


# ---------------------------------------------------------------- cached model fits
def indices():
    up = st.session_state.get("idx_uploads", {})
    return CI.load_indices(up)


@st.cache_data(show_spinner="Fitting market betas...")
def _beta():
    return CI.beta_alpha()


@st.cache_data(show_spinner="Fitting the should-cost model on 1,600 past bids...")
def _should(idx):
    sc = CI.fit_should_cost(idx)
    return sc


@st.cache_resource(show_spinner="Training the change-order text model...")
def _text_model():
    return CO.train_text_model()


def index_panel():
    idx, src = indices()
    with st.expander("Price indices used " + ("(all synthetic: upload real FRED / BLS series here)" if all(v == "synthetic" for v in src.values())
                                              else "(" + ", ".join(f"{k}: {v}" for k, v in src.items()) + ")")):
        st.markdown("The app ships with **synthetic** monthly indices so it runs anywhere. For real numbers, download each series "
                    "from FRED as CSV (Download → CSV) and drop it here; the app rebases it to Jan 2018 = 100.")
        st.dataframe(pd.DataFrame([{"index": k, "FRED series": v[0], "what it is": v[1], "in use": src[k]} for k, v in CI.FRED_SUGGEST.items()]),
                     hide_index=True, width="stretch")
        cols = st.columns(5)
        ups = dict(st.session_state.get("idx_uploads", {}))
        for c, k in zip(cols, CI.INDEX_COLS):
            f = c.file_uploader(k, type="csv", key=f"idx_{k}")
            if f is not None:
                ups[k] = pd.read_csv(f)
        if ups != st.session_state.get("idx_uploads", {}):
            st.session_state["idx_uploads"] = ups
            st.rerun()
        long = idx.melt(id_vars="period", value_vars=CI.INDEX_COLS, var_name="index", value_name="value")
        long["date"] = pd.PeriodIndex(long.period, freq="M").to_timestamp()
        ch = alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None), y=alt.Y("value:Q", title="Index (Jan 2018 = 100)"),
            color=alt.Color("index:N", scale=alt.Scale(domain=CI.INDEX_COLS, range=SERIES[:5]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["index", "period", alt.Tooltip("value:Q", format=".1f")]).properties(height=240)
        st.altair_chart(ch, width="stretch")
    return idx, src


# ---------------------------------------------------------------- Cost intelligence page
def cost_intel(ctx):
    back(ctx, "ci")
    hero("Cost intelligence · market and fair price", "Is it the market, or the contractor?",
         "Separates cost growth caused by material prices from growth specific to a contractor or project, tests whether contract "
         "rates are fair for the kind of work, and measures how much a price shock could cost the remaining work.")
    idx, src = index_panel()
    tabs = st.tabs(["Market vs contractor (beta / alpha)", "Should-cost", "What it costs today", "Cost-at-Risk"])

    # ---------- beta / alpha
    with tabs[0]:
        ba, tiers, hist = _beta()
        st.markdown("#### How strongly does each trade's cost follow its material prices?")
        k = st.columns(3)
        for c, r in zip(k, ba.itertuples()):
            c.metric(f"{r.trade} beta", f"{r.beta:.2f}", f"95% CI {r.beta_lo:.2f} to {r.beta_hi:.2f}", delta_color="off", delta_arrow="off",
                     help="Beta = % change in a package's cost for a 1% change in its trade's price index (like a stock's beta to the market).")
        c1, c2 = st.columns([3, 2])
        trade = c1.radio("Trade", list(ba.trade), horizontal=True, key="ba_trade")
        h = hist[hist.trade == trade]
        b = ba.set_index("trade").loc[trade]
        line = pd.DataFrame({"x": [h.index_change_pct.min(), h.index_change_pct.max()]})
        line["y"] = b.alpha_pct_month + b.beta * line.x
        sc = alt.Chart(h).mark_circle(size=34, opacity=0.55, color=SERIES[0]).encode(
            x=alt.X("index_change_pct:Q", title="Trade price index change in the month (%)"),
            y=alt.Y("cost_growth_pct:Q", title="Package cost growth in the month (%)"),
            tooltip=["package_id", "period", "contractor_tier", alt.Tooltip("index_change_pct:Q", format=".2f"), alt.Tooltip("cost_growth_pct:Q", format=".2f")])
        ln = alt.Chart(line).mark_line(color=SERIES[1], strokeWidth=2.5).encode(x="x:Q", y="y:Q")
        c1.caption(f"{trade}: {int(b.months)} package-months from {int(b.packages)} past packages")
        c1.altair_chart((sc + ln).properties(height=300), width="stretch")
        t = tiers[tiers.trade == trade].copy()
        t["lo"] = t.alpha_pct_month - 1.96 * t.alpha_se
        t["hi"] = t.alpha_pct_month + 1.96 * t.alpha_se
        bars = alt.Chart(t).mark_bar(height=22, cornerRadiusEnd=4, color=SERIES[2]).encode(
            y=alt.Y("tier:N", title=None), x=alt.X("alpha_pct_month:Q", title="Alpha: growth not explained by prices (% / month)"),
            tooltip=["tier", alt.Tooltip("alpha_pct_month:Q", format=".3f"), alt.Tooltip("alpha_p:Q", format=".3f", title="p-value"), "months"])
        err = alt.Chart(t).mark_rule(color="#e8eef8").encode(y="tier:N", x="lo:Q", x2="hi:Q")
        c2.altair_chart((bars + err).properties(height=220, title="Alpha by contractor tier (95% CI)"), width="stretch")
        howto(f"<b>How to read:</b> each dot is one past package in one month. The slope of the orange line is <b>beta</b>: for {trade.lower()} "
              f"work, cost grew {b.beta:.2f}% for every 1% rise in the trade's price index (R² {b.r2:.2f}). The intercept is <b>alpha</b>: "
              f"growth that happens even when prices are flat (scope creep, claims, productivity). Tier 3 contractors carry the largest "
              f"alpha. <b>Use it:</b> when a package overruns, the beta tells you how much the market explains; the rest is a conversation "
              f"with the contractor.")
        synthetic_note("History is a synthetic set of 60 past packages with known betas (0.33 to 0.42). The method is real; the numbers "
                       "become real when you load your own past cost reports.")

        st.markdown("#### This program: cost growth split into market and project-specific")
        split = CI.overrun_split(ctx.tables, idx, ba, ctx.end_period)
        if not len(split):
            st.info("Needs contractors mapped to a trade (electrical, mechanical, civil) and project start months: available for the "
                    "data center demo.")
        else:
            split["package"] = split.project_id.map(lambda p: ctx.labels.get(p, p)) + " · " + split.trade
            long = pd.concat([split.assign(part="Market (beta × index move)", v=split.market_usd),
                              split.assign(part="Project-specific (alpha)", v=split.specific_usd)])
            ch = alt.Chart(long).mark_bar(cornerRadiusEnd=3, stroke="#0e1525", strokeWidth=2).encode(
                y=alt.Y("package:N", sort=list(split.sort_values("specific_usd", ascending=False).package), title=None, axis=alt.Axis(labelLimit=220)),
                x=alt.X("v:Q", title="Cost growth vs original contract ($)", axis=alt.Axis(format="$~s")),
                color=alt.Color("part:N", scale=alt.Scale(domain=["Market (beta × index move)", "Project-specific (alpha)"], range=[SERIES[0], SERIES[1]]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["package", "part", alt.Tooltip("v:Q", format="$,.0f"), alt.Tooltip("index_move_pct:Q", format=".1f", title="Index move since award %"),
                         alt.Tooltip("growth_pct:Q", format=".1f", title="Total growth %")])
            st.altair_chart(ch.properties(height=alt.Step(24)), width="stretch")
            tot_m, tot_s = split.market_usd.sum(), split.specific_usd.sum()
            top = split.sort_values("specific_usd", ascending=False).iloc[0]
            howto(f"<b>Now:</b> change orders have added {money(tot_m + tot_s)} to the original contracts. About {money(tot_m)} "
                  f"({100 * tot_m / max(tot_m + tot_s, 1):.0f}%) is what the market move since award would predict; {money(tot_s)} is "
                  f"project-specific. Largest project-specific growth: <b>{top.package}</b> ({money(top.specific_usd)}). "
                  "Start the cost conversation there. <i>Caveat:</i> award month assumed one month before first billing; the split is an "
                  "estimate, not proof of fault.")

    # ---------- should-cost
    sc = _should(idx)
    with tabs[1]:
        st.markdown("#### Is this rate fair for this kind of work?")
        st.markdown("A **hedonic** regression prices each past bid from its attributes: the item, the price index when bid, the "
                    "quantity (bigger jobs get lower rates), region, union labour, an accelerated schedule and site difficulty. "
                    "The fitted model gives a *should-cost* and a range for any new rate.")
        comp = sc["comparison"].copy()
        comp["chosen"] = np.where(comp.model == sc["best"], "✓ lowest BIC", "")
        st.dataframe(comp, hide_index=True, width="stretch", column_config={
            "model": "Specification", "terms": "Terms", "adj_r2": st.column_config.NumberColumn("Adj. R²", format="%.3f"),
            "aic": st.column_config.NumberColumn("AIC", format="%.0f"), "bic": st.column_config.NumberColumn("BIC (SIC)", format="%.0f"),
            "resid_sd": st.column_config.NumberColumn("Residual sd (log)", format="%.3f"), "chosen": " "})
        m = sc["models"][sc["best"]]
        coef = m["fit"]["coef"]
        coef = coef[coef.term.isin(m["terms"])].copy()
        coef["label"] = coef.term.map(CI.TERM_LABEL)
        coef["effect_pct"] = np.where(coef.term.isin(["log_index", "log_qty_rel"]), np.nan, 100 * (np.exp(coef.estimate) - 1))
        coef["lo"], coef["hi"] = coef.estimate - 1.96 * coef.std_error, coef.estimate + 1.96 * coef.std_error
        c1, c2 = st.columns([3, 2])
        pts = alt.Chart(coef).mark_point(filled=True, size=80, color=SERIES[0]).encode(
            y=alt.Y("label:N", sort=list(coef.label), title=None, axis=alt.Axis(labelLimit=240)), x=alt.X("estimate:Q", title="Coefficient (log points, 95% CI)"),
            tooltip=["label", alt.Tooltip("estimate:Q", format=".3f"), alt.Tooltip("std_error:Q", format=".3f"), alt.Tooltip("p_value:Q", format=".2g"),
                     alt.Tooltip("effect_pct:Q", format="+.1f", title="Effect on rate %")])
        rules = alt.Chart(coef).mark_rule(color="#b9c3d3").encode(y=alt.Y("label:N", sort=list(coef.label)), x="lo:Q", x2="hi:Q")
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8792a6", strokeDash=[3, 3]).encode(x="x:Q")
        c1.altair_chart((rules + pts + zero).properties(height=alt.Step(26), title="What moves a rate (item effects not shown)"), width="stretch")
        with c2:
            st.markdown(f"**Do regions matter? (joint F-test)** F = {sc['region_F']:.1f}, p = {sc['region_p']:.1g} → "
                        + ("yes, keep the region dummies." if sc["region_p"] < 0.05 else "no clear evidence."))
            st.markdown("**Multicollinearity (VIF):** all below " + f"{sc['vif'].vif.max():.1f}" + " (below 5 is fine).")
            st.markdown("**Omitted-variable check:** adding site difficulty changes the other coefficients by "
                        f"{(sc['ovb'].with_site_difficulty - sc['ovb'].without_site_difficulty).abs().max():.3f} log points at most, "
                        "so leaving it out would not bias them here.")
        howto("<b>How to read:</b> a coefficient of 0.12 on union labour means union jobs bid about 13% higher, all else equal. The "
              "price-index coefficient is an elasticity: a 10% higher index means a ~" + f"{10 * coef.set_index('term').estimate.get('log_index', 0):.0f}"
              + "% higher rate. Quantity is negative: economies of scale.")

        st.markdown("#### This program's contract rates vs should-cost")
        if not len(sc["hist"]) or ctx.tables["po_lines"].item_code.isin(CI.FAMILY).sum() == 0:
            st.info("No bid history for these items. The demo history covers the data center rate card; load your own bid tabs "
                    "(item, date, region, quantity, rate) to price other work.")
        else:
            c = st.columns(5)
            reg = c[0].selectbox("Region", list(CI.REGIONS), key="sc_reg")
            uni = c[1].selectbox("Union labour", [0, 1], format_func=lambda v: "Yes" if v else "No", key="sc_uni")
            acc = c[2].selectbox("Accelerated schedule", [1, 0], format_func=lambda v: "Yes" if v else "No", key="sc_acc")
            dif = c[3].selectbox("Site difficulty", [1, 2, 3], key="sc_dif")
            lev = c[4].select_slider("Range", [0.8, 0.9, 0.95], value=0.9, format_func=lambda v: f"{v:.0%}", key="sc_lev")
            L = CI.should_cost_lines(sc, ctx.tables, idx, {"region": reg, "union_labor": uni, "accelerated": acc, "site_difficulty": dif}, level=lev)
            L = L[L.project_id.isin(ctx.pt.project_id)]
            L["project"] = L.project_id.map(lambda p: ctx.labels.get(p, p))
            k = st.columns(4)
            k[0].metric("Lines compared", len(L))
            k[1].metric("Above the fair range", int((L.flag == "Above range").sum()))
            k[2].metric("Paid above the range", money(L.excess_usd.sum()), help="(contract rate − top of range) × contract quantity, for lines above the range.")
            k[3].metric("Median gap vs should-cost", f"{L.gap_pct.median():+.1f}%")
            L["lab"] = L.project + " · " + L.family
            dots = alt.Chart(L).mark_circle(size=70, opacity=0.9).encode(
                x=alt.X("z:Q", title="Contract rate vs should-cost (standard errors; ±1.64 ≈ 90% range)"),
                y=alt.Y("family:N", title=None, axis=alt.Axis(labelLimit=200)),
                color=alt.Color("flag:N", scale=alt.Scale(domain=["Within range", "Above range", "Below range"], range=[SERIES[0], STATUS["critical"], SERIES[2]]),
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=["project", "family", alt.Tooltip("contract_unit_rate:Q", format="$,.2f", title="Contract rate"),
                         alt.Tooltip("should_cost:Q", format="$,.2f", title="Should-cost"), alt.Tooltip("pi_lo:Q", format="$,.2f", title="Range low"),
                         alt.Tooltip("pi_hi:Q", format="$,.2f", title="Range high"), alt.Tooltip("gap_pct:Q", format="+.1f", title="Gap %")])
            from scipy import stats as _st
            zc = _st.norm.ppf(0.5 + lev / 2)
            band = alt.Chart(pd.DataFrame({"a": [-zc], "b": [zc]})).mark_rect(opacity=0.08, color="#4fd1ff").encode(x="a:Q", x2="b:Q")
            st.altair_chart((band + dots).properties(height=alt.Step(22)), width="stretch")
            st.dataframe(L.sort_values("z", ascending=False)[["project", "family", "unit", "contract_qty", "contract_unit_rate", "should_cost", "pi_lo", "pi_hi",
                                                              "gap_pct", "flag", "excess_usd"]], hide_index=True, width="stretch", column_config={
                "project": "Project", "family": "Item", "contract_qty": st.column_config.NumberColumn("Qty", format="%,.0f"),
                "contract_unit_rate": st.column_config.NumberColumn("Contract rate", format="$%,.2f"),
                "should_cost": st.column_config.NumberColumn("Should-cost", format="$%,.2f"), "pi_lo": st.column_config.NumberColumn("Range low", format="$%,.2f"),
                "pi_hi": st.column_config.NumberColumn("Range high", format="$%,.2f"), "gap_pct": st.column_config.NumberColumn("Gap %", format="%+.1f"),
                "flag": "Verdict", "excess_usd": st.column_config.NumberColumn("Above range ($)", format="$%,.0f")})
            st.warning("**Guardrail (the Altreya lesson):** a regression can say a rate is *unusual*, not that it is *wrong*. The model cannot "
                       "see scope details, access or site constraints that are not in the data. Use 'Above range' to ask for a cost "
                       "breakdown, never to cut a payment.")
        synthetic_note("Bid history: 1,600 synthetic past bids built from known effects. Swap in your own bid tabs for real should-cost.")

    # ---------- today
    with tabs[2]:
        st.markdown("#### What would the same contract cost if we bought it today?")
        L0 = CI.should_cost_lines(sc, ctx.tables, idx)
        L0 = L0[L0.project_id.isin(ctx.pt.project_id)] if len(L0) else L0
        if not len(L0):
            st.info("Needs items mapped to a price index (data center demo).")
        else:
            L0["value_at_award"] = L0.contract_qty * L0.contract_unit_rate
            L0["value_today"] = L0.contract_qty * L0.rate_today
            fam = L0.groupby("family").agg(award=("value_at_award", "sum"), today=("value_today", "sum")).reset_index()
            fam["change_pct"] = 100 * (fam.today / fam.award - 1)
            k = st.columns(3)
            k[0].metric("Unit-price work at award", money(L0.value_at_award.sum()))
            k[1].metric("Same work at today's prices", money(L0.value_today.sum()), f"{100 * (L0.value_today.sum() / L0.value_at_award.sum() - 1):+.1f}%",
                        delta_color="inverse")
            k[2].metric("Latest index month", idx.period.iloc[-1])
            ch = alt.Chart(fam).mark_bar(height=18, cornerRadiusEnd=4).encode(
                y=alt.Y("family:N", sort="-x", title=None), x=alt.X("change_pct:Q", title="Price change since award (%)"),
                color=alt.condition(alt.datum.change_pct > 0, alt.value(SERIES[1]), alt.value(SERIES[2])),
                tooltip=["family", alt.Tooltip("award:Q", format="$,.0f"), alt.Tooltip("today:Q", format="$,.0f"), alt.Tooltip("change_pct:Q", format="+.1f")])
            st.altair_chart(ch.properties(height=alt.Step(24)), width="stretch")
            howto("<b>How it works:</b> each rate is split into a material share (moves with its price index: steel, copper, concrete or "
                  "equipment) and a labour share (moves with construction wages), then escalated from the award month to the latest index "
                  "month. <b>Use it:</b> it tells you whether locked-in contracts are now a bargain (prices rose) or expensive (prices fell), "
                  "which matters for change-order pricing and the next bid round.")

    # ---------- VaR
    with tabs[3]:
        st.markdown("#### How much could material prices add to the remaining work?")
        c = st.columns(3)
        pt_ = c[0].slider("Share of price risk the owner carries", 0.0, 1.0, 0.5, 0.05, key="var_pass",
                          help="Escalation clauses and change orders pass some price risk to the owner. 0 = fully fixed-price contracts.")
        hz = c[1].select_slider("Horizon (months)", [3, 6, 9, 12], value=6, key="var_h")
        cf = c[2].select_slider("Confidence", [0.9, 0.95, 0.99], value=0.95, format_func=lambda v: f"{v:.0%}", key="var_c")
        ex = CI.exposure(ctx.pt, ctx.tables, pt_)
        ex = ex[ex.project_id.isin(ctx.pt.project_id)] if len(ex) else ex
        if not len(ex) or ex.exposure.sum() <= 0:
            st.info("Needs contractors mapped to a trade to size material exposure (data center demo).")
            return
        v = CI.var_report(idx, ex, hz, cf)
        k = st.columns(4)
        k[0].metric(f"Cost-at-Risk (analytical, {cf:.0%})", money(v["analytical"]), help="Normal approximation: mean + z × sd × √horizon.")
        k[1].metric("Historical", money(v["historical"]), help="Worst rolling windows of the same length in the index history.")
        k[2].metric("Monte Carlo", money(v["monte_carlo"]), help="20,000 simulated paths, resampling 3-month blocks of real joint index moves.")
        k[3].metric("Expected shortfall (beyond VaR)", money(v["cvar"]), help="Average cost in the worst tail beyond the Monte Carlo VaR.")
        normal_ok = v["jb_p"] > 0.05
        st.markdown(f"**Normality check (Jarque-Bera):** p = {v['jb_p']:.3f}, skew {v['skew']:+.2f}, excess kurtosis {v['kurtosis']:+.2f} → "
                    + ("the analytical (normal) figure is usable." if normal_ok else "**not normal**: prefer the historical or Monte Carlo figure."))
        c1, c2 = st.columns(2)
        sims = pd.DataFrame({"impact": v["sims"] / 1e6})
        hist_c = alt.Chart(sims).mark_bar(color=SERIES[0], opacity=0.85).encode(
            x=alt.X("impact:Q", bin=alt.Bin(maxbins=50), title=f"Price impact over {hz} months ($M)"), y=alt.Y("count():Q", title="Paths"))
        vr = alt.Chart(pd.DataFrame({"x": [v["monte_carlo"] / 1e6]})).mark_rule(color=STATUS["critical"], strokeDash=[4, 3], strokeWidth=2).encode(x="x:Q")
        c1.altair_chart((hist_c + vr).properties(height=260, title="Monte Carlo: 20,000 price paths (red = VaR)"), width="stretch")
        stx = v["stress"].sort_values("cost_impact", ascending=False)
        sb = alt.Chart(stx).mark_bar(height=18, cornerRadiusEnd=4, color=SERIES[1]).encode(
            y=alt.Y("scenario:N", sort=list(stx.scenario), title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X("cost_impact:Q", title="Cost impact ($)", axis=alt.Axis(format="$~s")), tooltip=["scenario", alt.Tooltip("cost_impact:Q", format="$,.0f")])
        c2.altair_chart(sb.properties(height=alt.Step(30), title="Stress scenarios (one-off shocks)"), width="stretch")
        e = v["exposure_by_index"]
        howto(f"<b>How to read:</b> the owner carries {money(e.sum())} of material price exposure on the unbilled work "
              f"(largest: {e.idxmax()}, {money(e.max())}). In {cf:.0%} of {hz}-month futures, price moves add less than "
              f"{money(v['monte_carlo'])}; that is the contingency to hold for prices. Stress scenarios show what VaR cannot: a repeat of the 2021 "
              f"steel surge would cost {money(stx.cost_impact.max())}. <b>Decisions:</b> lock prices early (buy steel and copper ahead), add "
              "escalation caps to new contracts, size contingency to the larger of VaR and the stress you cannot accept.")
        synthetic_note("Index history is synthetic unless you upload real FRED series above; exposure uses assumed material shares "
                       "(electrical 45%, mechanical 50%, civil 40%).")


# ---------------------------------------------------------------- Change orders page
def change_orders(ctx):
    back(ctx, "co")
    hero("Change orders · decide with evidence", "Approve, negotiate or reject?",
         "Every change order is read, priced and run through a decision tree. The analyst flags weak justifications, rates above the "
         "base contract and scope the base contract may already cover. The project manager makes the final call.")
    model = _text_model()
    det = ctx.extras.get("co_details")
    idx, _ = indices()
    sc = _should(idx)
    should = CI.should_cost_lines(sc, ctx.tables, idx)
    with st.expander("Decision assumptions (edit to test your own view)"):
        a = dict(CO.DEFAULTS)
        c = st.columns(4)
        a["p_negotiate"] = c[0].slider("Chance negotiation lowers the price", 0.0, 1.0, a["p_negotiate"], 0.05)
        a["settle_frac"] = c[1].slider("Settled price (% of quote)", 0.6, 1.0, a["settle_frac"], 0.01)
        a["day_cost"] = c[2].number_input("Owner's cost of one day of delay ($)", 0.0, 1e6, a["day_cost"], 5000.0,
                                          help="Lost revenue, rented power, extended general conditions. Data centers: often $25K-$100K/day.")
        a["claim_mult"] = c[3].slider("Claim cost if we reject and lose (× quote)", 1.0, 2.0, a["claim_mult"], 0.05)
        c = st.columns(4)
        a["legal_cost"] = c[0].number_input("Legal / dispute cost ($)", 0.0, 5e5, a["legal_cost"], 5000.0)
        a["negotiate_days"] = c[1].slider("Negotiation time (days)", 0, 30, int(a["negotiate_days"]))
        a["schedule_share"] = c[2].slider("Share of negotiation time that hits the schedule", 0.0, 1.0, a["schedule_share"], 0.05,
                                          help="Applies only when the change is on the critical path (schedule impact > 0).")
        a["claim_slope"] = c[3].slider("How much a strong justification raises claim risk", 0.0, 0.8, a["claim_slope"], 0.05)
        st.caption("Chance the contractor claims and wins if we reject = 15% + slope × P(justified).")
        use_sched = st.toggle("Use the schedule (time impact analysis) for delay days", False, key="co_tia",
                              help="Off: the delay a contractor claims if rejected. On: that delay inserted into the CPM network; float absorbs part of it.")
    delay_fn = None
    if use_sched:
        from ui.schedule_page import co_delay_fn
        delay_fn = co_delay_fn(ctx)
    A = CO.analyze(ctx.tables, det, model, a, should, delay_fn)
    A = A[A.project_id.isin(ctx.pt.project_id)]
    if not len(A):
        st.info("No change orders in this view.")
        return
    A["project"] = A.project_id.map(lambda p: ctx.labels.get(p, p))
    A["contractor"] = A.prime_id.map(lambda p: ctx.names.get(p, p))
    pend = A[A.status == "PENDING"]
    k = st.columns(4)
    k[0].metric("Pending change orders", f"{len(pend)}", money(pend.amount.sum()), delta_color="off", delta_arrow="off")
    k[1].metric("With at least one concern", int((pend.concerns != "none found").sum()))
    k[2].metric("Recommended: reject", int((pend.recommendation == "Reject").sum()))
    k[3].metric("Expected saving vs approving all", money(pend.expected_saving.sum()),
                help="Sum over pending change orders of (cost of approving as priced − expected cost of the recommended option).")
    show = st.radio("Show", ["Pending only", "All change orders"], horizontal=True, key="co_show")
    T = pend if show == "Pending only" else A
    st.dataframe(T[["co_id", "project", "contractor", "status", "amount", "p_justified", "rate_premium_pct", "base_used_pct", "recommendation",
                    "expected_saving", "concerns"]].sort_values("expected_saving", ascending=False), hide_index=True, width="stretch", column_config={
        "co_id": "Change order", "project": "Project", "contractor": "Contractor", "status": "Status",
        "amount": st.column_config.NumberColumn("Amount", format="$%,.0f"),
        "p_justified": st.column_config.ProgressColumn("Justified?", min_value=0, max_value=1, format="percent"),
        "rate_premium_pct": st.column_config.NumberColumn("Rate vs base %", format="%+.1f"),
        "base_used_pct": st.column_config.NumberColumn("Base qty billed %", format="%.0f"), "recommendation": "Recommendation",
        "expected_saving": st.column_config.NumberColumn("Expected saving", format="$%,.0f"), "concerns": "Concerns"})
    if not len(T):
        return
    st.markdown("### Change order analyst")
    pick = st.selectbox("Change order", list(T.co_id), format_func=lambda c: f"{c} · {A.set_index('co_id').project[c]} · {money(A.set_index('co_id').amount[c])}",
                        key="co_pick")
    r = A.set_index("co_id").loc[pick]
    c1, c2 = st.columns([2, 3])
    with c1, st.container(border=True):
        st.markdown(f"**{pick}** · {r.project} · {r.contractor}  \n**{money(r.amount)}** · status {r.status}")
        st.markdown(f"> {r.justification}")
        st.progress(float(r.p_justified), text=f"Reads as owner-caused / justified: {r.p_justified:.0%}")
        if r.positive_terms:
            st.markdown(f":green[Supporting words:] {r.positive_terms}")
        if r.negative_terms:
            st.markdown(f":orange[Contractor-side words:] {r.negative_terms}")
        st.markdown(f"**Rate vs base contract:** {'n/a' if pd.isna(r.rate_premium_pct) else f'{r.rate_premium_pct:+.1f}%'}  \n"
                    f"**Base quantity billed so far:** {'n/a' if pd.isna(r.base_used_pct) else f'{r.base_used_pct:.0f}%'}  \n"
                    f"**Schedule impact if rejected:** {r.delay_days} days ({r.delay_basis})  \n**Concerns:** {r.concerns}")
    with c2:
        d = CO.decision(r.amount, r.p_justified, r.delay_days, a)
        st.graphviz_chart(CO.tree_dot(d, r.amount), width="stretch")
        dom = (" Negotiate dominates " + " and ".join(d["dominated"]) + " (cheaper on average and in the worst case).") if d["dominated"] else ""
        howto(f"<b>Expected cost:</b> approve {money(d['ev']['Approve as priced'])}, negotiate {money(d['ev']['Negotiate'])}, "
              f"reject {money(d['ev']['Reject'])}. <b>Lowest expected cost: {d['best']}.</b>{dom} If rejected, the chance the contractor claims and wins is "
              f"{d['q_claim']:.0%} (from the justification score).")
        st.caption("Scenario analysis, not a recommendation to reject. First ask whether the contractor is entitled under the "
                   "contract; if so, the change is owed whatever the model says. The claim odds are assumptions, and the decision "
                   "and its reason are recorded by a person.")
    # sensitivity: EV vs P(justified)
    ps = np.linspace(0, 1, 41)
    sens = pd.DataFrame([{"p": p, "option": o, "ev": v} for p in ps for o, v in CO.decision(r.amount, p, r.delay_days, a)["ev"].items()])
    lc = alt.Chart(sens).mark_line(strokeWidth=2).encode(
        x=alt.X("p:Q", title="Probability the change is justified", axis=alt.Axis(format="%")), y=alt.Y("ev:Q", title="Expected cost ($)", axis=alt.Axis(format="$~s")),
        color=alt.Color("option:N", scale=alt.Scale(domain=["Approve as priced", "Negotiate", "Reject"], range=[SERIES[0], SERIES[2], SERIES[1]]),
                        legend=alt.Legend(orient="top", title=None)), tooltip=["option", alt.Tooltip("p:Q", format=".0%"), alt.Tooltip("ev:Q", format="$,.0f")])
    now = alt.Chart(pd.DataFrame({"x": [r.p_justified]})).mark_rule(color="#e8eef8", strokeDash=[3, 3]).encode(x="x:Q")
    st.altair_chart((lc + now).properties(height=240, title="Sensitivity: which option is cheapest as the justification gets stronger"), width="stretch")
    be = CO.breakeven_p(r.amount, r.delay_days, a)
    howto("<b>How to read:</b> the lowest line is the cheapest option at that level of justification; the dashed line is this change "
          "order's score. " + (f"Rejecting stops paying off once the justification is stronger than {be:.0%}." if 0 < be < 1 else
                               "Rejecting is never the cheapest option here: the schedule cost of a dispute is too high." if be == 0 else ""))
    st.markdown("#### Memo")
    memo = CO.memo_template(r.to_dict() | {"co_id": pick}, r.project, r.contractor)
    if ai.available():
        if st.button(f"Draft with Gemini ({ai.model_name()})", key=f"memo_ai_{pick}"):
            txt, err = ai.generate(CO.memo_prompt(r.to_dict() | {"co_id": pick}, r.project, r.contractor))
            st.session_state[f"memo_{pick}"] = txt if txt else f"_{err}_\n\n" + memo
    else:
        st.caption("Template memo. Add GEMINI_API_KEY in the app's secrets to have Gemini draft it.")
    memo = st.session_state.get(f"memo_{pick}", memo)
    with st.container(border=True):
        st.markdown(memo)
    st.download_button("Download memo", memo, f"{pick}_memo.md", key=f"dl_{pick}")
    with st.expander("How the text model works"):
        st.markdown(f"Naive Bayes on word pairs, calibrated, trained on {model['n']} labelled justifications (synthetic). "
                    f"5-fold cross-validated accuracy **{model['cv_accuracy']:.0%}** (± {model['cv_sd']:.0%}).")
        w = model["words"]
        top = pd.concat([w.nlargest(8, "log_odds_justified"), w.nsmallest(8, "log_odds_justified")])
        ch = alt.Chart(top).mark_bar(height=14, cornerRadiusEnd=3).encode(
            y=alt.Y("term:N", sort="-x", title=None), x=alt.X("log_odds_justified:Q", title="Log-odds toward 'justified'"),
            color=alt.condition(alt.datum.log_odds_justified > 0, alt.value(SERIES[2]), alt.value(SERIES[1])))
        st.altair_chart(ch.properties(height=alt.Step(18)), width="stretch")
        synthetic_note("Text training set and change-order justifications are synthetic. Retrain on your own reviewed change orders.")
