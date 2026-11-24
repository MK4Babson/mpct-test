"""Global markets & trade: the three macro markets for each sourcing country, a central-bank shock lab (M1, M2/M3,
cash, credit, FX), balance of payments and reserves, local labor, tariffs and landed cost, FX exposure, the ten-driver
supplier-country risk register, and a ledger of what a shock costs this program inside its window."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import macro as X
import schedule as S
from insights import money
from ui.cost_pages import back, hero, howto, synthetic_note
from ui.scm_pages import DATA_DATE, model as scm_model
from ui.theme import SERIES, STATUS

DOLLAR = st.column_config.NumberColumn(format="$%,.0f")
NUM = st.column_config.NumberColumn(format="%,.2f")
PCT = st.column_config.NumberColumn(format="%.2f%%")
LEVEL_COLOR = alt.Scale(domain=["Low", "Medium", "High"], range=[STATUS["good"], STATUS["warning"], STATUS["critical"]])


@st.cache_data(show_spinner=False)
def _data():
    return X.load()


def _months_left(ctx):
    pc = ctx.controls.get("project_controls") if getattr(ctx, "controls", None) else None
    if pc is not None and len(pc) and "planned_finish" in pc:
        fin = pd.to_datetime(pc.planned_finish).max()
        return max(1, int(round((fin - DATA_DATE).days / 30.4)))
    return 12


def page(ctx):
    back(ctx, "gm")
    hero("Global markets & trade", "What the world's three big markets are doing, and what it costs this program",
         "Credit (interest rates, money), goods and services (prices, output, jobs) and foreign exchange, for the US and every country "
         "we buy from, plus the balance of payments, reserves, local labor, tariffs and FX exposure. Every shock ends in dollars inside "
         "the project window.")
    synthetic_note("Synthetic macro, labor and BOP series (realistic shapes, not real figures). Tariff rates are labelled 'from source' or "
                   "'ILLUSTRATIVE'. Real data: FRED, RBI DBIE, IMF BOP, BLS LAUS/CES/OEWS in the same columns (see Method & limits).")
    d = _data()
    tabs = st.tabs(["Three markets", "Central bank shock lab", "Balance of payments & reserves", "Local labor", "Tariffs & landed cost",
                    "FX exposure", "Supplier-country risk", "Project impact ledger", "Method & limits"])
    with tabs[0]:
        _markets(d)
    with tabs[1]:
        _lab(d, ctx)
    with tabs[2]:
        _bop(d)
    with tabs[3]:
        _labor(d, ctx)
    with tabs[4]:
        _tariffs(d, ctx)
    with tabs[5]:
        _fx(d, ctx)
    with tabs[6]:
        _risk(d, ctx)
    with tabs[7]:
        _ledger(d, ctx)
    with tabs[8]:
        _method()


def _country_pick(d, key, default="IN"):
    c = d["countries"]
    lab = dict(zip(c.code, c.name + " (" + c.currency + ")"))
    return st.selectbox("Country", list(lab), index=list(lab).index(default), format_func=lab.get, key=key)


# ---------------------------------------------------------------- tabs
def _markets(d):
    howto("Pick a country. <b>Credit market</b>: policy rate, the rate firms pay, and the real rate (exact Fisher: (1+i)/(1+expected inflation) − 1). "
          "<b>Goods market</b>: inflation, real GDP growth, unemployment. <b>FX market</b>: currency per USD and the real effective exchange rate "
          "(REER: competitiveness after prices; up = dearer). Money: M1 and broad money growth.")
    code = _country_pick(d, "mk_c")
    mon, ctry = d["monetary"], d["countries"].set_index("code").loc[code]
    x = mon[mon.country == code].sort_values("month").copy()
    x["date"] = x.month.dt.to_timestamp()
    x["real_rate"] = X.fisher_real(x.lending_rate, x.expected_inflation)
    x["M1_yoy"] = 100 * (x.M1 / x.M1.shift(12) - 1)
    x["broad_yoy"] = 100 * (x.broad_money / x.broad_money.shift(12) - 1)
    last = x.iloc[-1]
    st.caption(f"{ctry['name']} · {ctry.central_bank} · {ctry.framework} · FX regime: {ctry.fx_regime} · capital mobility: {ctry.capital_mobility} · "
               f"broad money reported as {ctry.broad_label}")
    k = st.columns(6)
    k[0].metric("Policy rate", f"{last.policy_rate:.2f}%")
    k[1].metric("Lending rate", f"{last.lending_rate:.2f}%")
    k[2].metric("Real lending rate", f"{last.real_rate:.2f}%", help="Exact Fisher with expected inflation")
    k[3].metric("Inflation", f"{last.cpi_yoy:.1f}%", f"expected {last.expected_inflation:.1f}%", delta_color="off")
    k[4].metric("Real GDP growth", f"{last.real_gdp_yoy:.1f}%", f"unemployment {last.unemployment:.1f}%", delta_color="off")
    k[5].metric(f"{ctry.currency} per USD" if code != "US" else "REER (USD)", f"{last.fx_per_usd:,.2f}" if code != "US" else f"{last.reer:.0f}",
                f"REER {last.reer:.0f}" if code != "US" else None, delta_color="off")

    def lines(cols, title, fmt="%"):
        m = x.melt(id_vars="date", value_vars=list(cols), var_name="series", value_name="v")
        m["series"] = m.series.map(cols)
        return alt.Chart(m).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None), y=alt.Y("v:Q", title=None),
            color=alt.Color("series:N", scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["date:T", "series:N", alt.Tooltip("v:Q", format=",.2f")]).properties(height=220, title=title)
    c1, c2 = st.columns(2)
    c1.altair_chart(lines({"policy_rate": "Policy rate", "lending_rate": "Lending rate", "real_rate": "Real lending rate"}, "Credit market (%)"), width="stretch")
    c2.altair_chart(lines({"cpi_yoy": "Inflation", "real_gdp_yoy": "Real GDP growth", "unemployment": "Unemployment"}, "Goods & services market (%)"), width="stretch")
    c1, c2 = st.columns(2)
    c1.altair_chart(lines({"M1_yoy": "M1 growth", "broad_yoy": f"{ctry.broad_label} growth"}, "Money (% a year)"), width="stretch")
    if code != "US":
        c2.altair_chart(lines({"fx_per_usd": f"{ctry.currency} per USD"}, "FX market: nominal rate"), width="stretch")
    c2.altair_chart(lines({"reer": "REER (index)"}, "FX market: real effective exchange rate"), width="stretch")


def _lab(d, ctx):
    howto("Set a central-bank move and see the chain, as in the three-sector model: <b>money</b> (base, cash, deposits, M1, broad money, "
          "multiplier), <b>credit</b> (lending and real rates), <b>FX</b> (forwards on our open orders) and the <b>goods market</b> lag. "
          "Direct effects (the base) outweigh indirect ones (the multiplier). The method depends on how the central bank actually moves rates.")
    c1, c2, c3 = st.columns(3)
    with c1:
        code = _country_pick(d, "lab_c")
    ctry = d["countries"].set_index("code").loc[code]
    dr = c2.slider("Policy rate change (percentage points)", -2.0, 2.0, 0.5, 0.25, key="lab_dr")
    modes = {"drain": "Drain/add reserves (open-market or liquidity operations)", "rate_only": "Move the rate paid on reserves (floor system)",
             "reserve_ratio": "Change the reserve requirement"}
    default = "rate_only" if ctry.framework.startswith("floor") else "drain"
    mode = c3.selectbox("How it is done", list(modes), index=list(modes).index(default), format_func=modes.get, key="lab_mode")
    a, b, c = st.columns(3)
    drain = a.slider("Base change per +1pp (drain mode)", 0.0, 0.15, 0.06, 0.01, disabled=mode != "drain", format="%.2f",
                     help="Share of the monetary base withdrawn for each +1pp. Assumption: set from the central bank's actual operations.")
    drr = b.slider("Reserve requirement change (pp)", -0.02, 0.02, 0.005, 0.0025, disabled=mode != "reserve_ratio", format="%.4f")
    loan_e = c.slider("Loan demand: % change per +1pp lending rate (floor mode)", -0.06, 0.0, -0.02, 0.005, disabled=mode != "rate_only", format="%.3f")
    mon = d["monetary"]
    row = X.latest(mon, code, DATA_DATE)
    sens = X.sensitivities(mon, code)
    pt, pt_src = X.pass_through(mon, code)
    t, dec = X.money_shock(row, ctry, sens, dr, mode, drain, drr, loan_e, pt)
    st.markdown("##### 1 · Money")
    show = t.loc[["monetary_base", "currency", "checkable_deposits", "near_money", "excess_reserves", "M1", "broad_money", "mm2"]].copy()
    show.index = ["Monetary base (B)", "Cash with the public (CC)", "Checkable deposits (D)", "Savings & time deposits (N)", "Banks' excess reserves (ER)",
                  "M1", f"Broad money ({ctry.broad_label})", f"Multiplier ({ctry.broad_label} / B)"]
    c1, c2 = st.columns([3, 2])
    c1.dataframe(show, width="stretch", column_config={"before": NUM, "after": NUM, "change_%": PCT})
    bars = show.drop(index="Banks' excess reserves (ER)").reset_index().rename(columns={"index": "aggregate"})
    c2.altair_chart(alt.Chart(bars).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("change_%:Q", title="% change"), y=alt.Y("aggregate:N", sort=list(show.index), title=None, axis=alt.Axis(labelLimit=240)),
        color=alt.condition(alt.datum["change_%"] < 0, alt.value(SERIES[1]), alt.value(SERIES[0])),
        tooltip=["aggregate", alt.Tooltip("change_%:Q", format="+.2f")]).properties(height=alt.Step(26)), width="stretch")
    if mode == "rate_only" and not ctry.framework.startswith("floor"):
        st.caption("Note: this country runs a corridor system, so 'rate on reserves only' is a what-if. Excess reserves become the residual and can swing a lot from a small base.")
    if mode == "rate_only":
        st.caption("Floor system: the base does not move and reserves are not the constraint. Total money follows loan demand (loans create "
                   "deposits); the public's ratios change only the mix. Banks keep reserves because reserves now pay more too.")
    else:
        st.caption(f"Broad money = multiplier × base. Direct (base) effect {dec['direct_base_%']:+.2f}%, indirect (multiplier) effect "
                   f"{dec['indirect_multiplier_%']:+.2f}%.")
    s_tbl = pd.DataFrame({"ratio": ["CC/D (cash per deposit)", "N/D (time deposits per deposit)", "ER/D (idle reserves per deposit)"],
                          "log-change per +1pp": [sens[k][0] for k in "cne"], "source": [sens[k][1] for k in "cne"]})
    with st.expander("Behaviour estimated from history (preferred asset ratios)"):
        st.dataframe(s_tbl, hide_index=True, width="stretch")
    st.markdown("##### 2 · Credit market")
    lend0, infl = row.lending_rate, row.expected_inflation
    lend1 = lend0 + pt * dr
    k = st.columns(4)
    k[0].metric("Pass-through to lending rates", f"{pt:.2f}", pt_src, delta_color="off")
    k[1].metric("Lending rate", f"{lend1:.2f}%", f"{pt * dr:+.2f}pp")
    k[2].metric("Real lending rate (exact Fisher)", f"{X.fisher_real(lend1, infl):.2f}%", f"from {X.fisher_real(lend0, infl):.2f}%", delta_color="off")
    k[3].metric("Banks' lending capacity", f"{dec['lending_capacity_%']:+.2f}%")
    st.caption("Expected inflation is held fixed here: in the book's model it does not change the real rate, only the nominal one.")
    st.markdown("##### 3 · FX market")
    scm = scm_model(ctx)
    op = scm["d"]["open_pos"]
    ex0 = X.fx_exposure(op, mon, DATA_DATE)
    ex1 = X.fx_exposure(op, mon, DATA_DATE, rate_shock=(code, dr))
    if len(ex0):
        cur = [c_ for c_ in ex0.currency.unique() if X.CURRENCY_COUNTRY[c_] == code] if code != "US" else list(ex0.currency.unique())
        e0, e1 = ex0[ex0.currency.isin(cur)], ex1[ex1.currency.isin(cur)]
        if len(e0):
            k = st.columns(3)
            k[0].metric("Open orders affected (USD at spot)", money(e0.usd_at_spot.sum()))
            k[1].metric("Hedged cost before", money(e0.usd_hedged_forward.sum()))
            k[2].metric("Hedged cost after the move", money(e1.usd_hedged_after_shock.sum()), money(e1.usd_hedged_after_shock.sum() - e0.usd_hedged_forward.sum()),
                        delta_color="inverse")
            st.caption("Forward = spot × (1 + foreign rate)/(1 + US rate)^t moves the same day. A higher foreign rate makes buying that currency "
                       "forward cheaper, but if the currency strengthens on the news (assumed 2% per +1pp) the dollar cost rises.")
        else:
            st.caption("No open orders in this currency.")
    st.markdown("##### 4 · Goods market (with a lag)")
    ml = _months_left(ctx)
    w = X.lag_weights().reset_index()
    w.columns = ["month", "weight"]
    w["inside"] = np.where(w.month <= ml, "inside the project window", "after the project")
    st.altair_chart(alt.Chart(w).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("month:O", title="Months after the rate change", axis=alt.Axis(labelAngle=0)), y=alt.Y("weight:Q", title=None, axis=alt.Axis(format=".0%")),
        color=alt.Color("inside:N", scale=alt.Scale(domain=["inside the project window", "after the project"], range=[SERIES[0], "#3b4a66"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["month", alt.Tooltip("weight:Q", format=".1%"), "inside"]).properties(height=200, title="Share of the effect arriving each month"), width="stretch")
    st.caption(f"Spending, prices and output respond over months (lag assumed to peak at 12 months). With {ml} months left, "
               f"{100 * X.share_in_window(ml):.0f}% of the effect lands inside the program. Dollar effects: Project impact ledger tab.")


def _bop(d):
    howto("The identity <b>CA + KA + NEO = FA + RRI</b> always holds (net errors and omissions make it balance). FA > 0: the country lends to the "
          "world. RRI > 0: the central bank bought foreign currency (adds to the monetary base). Warning signs for a supplier's currency: "
          "current-account deficit, reserves under 3 months of imports, reserves below short-term external debt (Greenspan-Guidotti), high "
          "inflation, an overvalued REER, a managed currency losing reserves.")
    ew = X.early_warning(d["bop"], d["monetary"], d["countries"], DATA_DATE)
    st.dataframe(ew, hide_index=True, width="stretch",
                 column_config={"ca_%gdp": PCT, "reserves_usd_bn": NUM, "import_cover_months": NUM, "greenspan_guidotti": NUM, "inflation_gap_pp": NUM,
                                "reer_vs_10y_avg_%": PCT})
    code = _country_pick(d, "bop_c")
    b = X.bop_check(d["bop"])
    x = b[b.country == code]
    st.dataframe(x[["year", "current_account", "capital_account", "net_errors_omissions", "financial_account", "reserves_related_items", "lhs_CA_KA_NEO",
                    "rhs_FA_RRI", "identity_holds", "net_lender", "central_bank"]], hide_index=True, width="stretch",
                 column_config={c: NUM for c in ["current_account", "capital_account", "net_errors_omissions", "financial_account", "reserves_related_items", "lhs_CA_KA_NEO", "rhs_FA_RRI"]})
    m = x.melt(id_vars="year", value_vars=["current_account", "financial_account", "reserves_related_items"], var_name="account", value_name="usd_bn")
    st.altair_chart(alt.Chart(m).mark_bar().encode(x=alt.X("year:O", title=None), xOffset="account:N", y=alt.Y("usd_bn:Q", title="USD bn"),
                                                   color=alt.Color("account:N", scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="bottom", title=None)),
                                                   tooltip=["year", "account", alt.Tooltip("usd_bn:Q", format=",.1f")]).properties(height=220), width="stretch")
    st.markdown("##### Intervention → monetary base")
    ctry = d["countries"].set_index("code").loc[code]
    fx = X.latest(d["monetary"], code, DATA_DATE).fx_per_usd
    a, b2 = st.columns(2)
    sold = a.number_input("Central bank sells FX reserves (USD bn) to support its currency", 0.0, 500.0, 10.0, 1.0)
    ster = b2.slider("Share offset by bond purchases (sterilized)", 0.0, 1.0, 0.0, 0.1)
    dB = X.intervention_to_base(sold, fx, ster)
    st.metric(f"Change in {ctry['name']}'s monetary base ({ctry.currency} bn)", f"{dB:,.0f}")
    st.caption("Selling dollars pulls domestic currency out of the banking system (Venezuela 2015, the Bank of Japan's yen support). Buying dollars "
               "adds it (the Swiss cap in 2011-15). Without sterilization the base, and broad money, move.")


def _labor(d, ctx):
    howto("Local craft labor is often the real constraint on a data center. The <b>tightness index</b> (0-100) combines unemployment vs its own "
          "history, construction job growth, electrician wage growth, and electricians wanted by competing projects vs those available. "
          "High tightness widens the pessimistic durations of labor-heavy tasks and adds a wage allowance.")
    lt = X.labor_tightness(d["labor_metro"], d["competing_projects"], d["trade_supply"], DATA_DATE)
    st.altair_chart(alt.Chart(lt).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("tightness_index:Q", title="Tightness index", scale=alt.Scale(domain=[0, 100])), y=alt.Y("metro:N", sort="-x", title=None),
        color=alt.Color("level:N", scale=LEVEL_COLOR, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["metro", alt.Tooltip("tightness_index:Q", format=".0f"), "level"]).properties(height=alt.Step(28)), width="stretch")
    st.dataframe(lt, hide_index=True, width="stretch", column_config={c: NUM for c in lt.columns if lt[c].dtype.kind == "f"})
    lm = d["labor_metro"].copy()
    lm["date"] = lm.month.dt.to_timestamp()
    c1, c2 = st.columns(2)
    for col, title, box in [("unemployment", "Unemployment (%)", c1), ("electrician_hourly", "Electrician wage ($/hour)", c2)]:
        box.altair_chart(alt.Chart(lm).mark_line(strokeWidth=2).encode(x=alt.X("date:T", title=None), y=alt.Y(f"{col}:Q", title=None),
                                                                        color=alt.Color("metro:N", scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="bottom", title=None)))
                         .properties(height=200, title=title), width="stretch")
    st.markdown("**Competing projects hiring the same trades**")
    st.dataframe(d["competing_projects"], hide_index=True, width="stretch")
    st.markdown("##### Turn tightness into schedule and cost numbers")
    from ui import schedule_page as SP
    pids = list(ctx.pt.project_id)
    c1, c2 = st.columns(2)
    pid = c1.selectbox("Building", pids, key="lab_pid")
    metro = c2.selectbox("Labor market it draws on", list(lt.metro), index=list(lt.metro).index("Odessa") if "Odessa" in list(lt.metro) else 0, key="lab_metro")
    r = lt.set_index("metro").loc[metro]
    ml = _months_left(ctx)
    adj = X.labor_adjustments(r.tightness_index, r["electrician_wage_yoy_%"], ml)
    k = st.columns(3)
    k[0].metric("Widen pessimistic durations (labor-heavy tasks)", f"+{100 * adj['pessimistic_widening']:.0f}%")
    k[1].metric(f"Wage allowance over {ml} months", f"{100 * adj['wage_allowance']:.1f}%")
    try:
        t, meta = SP.network(ctx, pid)
    except Exception:
        t = None
    if t is None:
        st.info("The schedule test needs project_controls with planned dates for this building.")
        return
    labor_trades = {"Electrical", "Mechanical", "Commissioning"}
    t2 = t.copy()
    mask = t2.trade.isin(labor_trades) & (t2.kind == "work")
    t2.loc[mask, "p"] = t2.loc[mask, "p"] * (1 + adj["pessimistic_widening"])
    base = SP._mc(t, meta, ctx.source_key, pid, ctx.end_period)
    tight = S.monte_carlo(t2, meta, 3000)
    k[2].metric("P80 finish moves by", f"{tight['p80'] - base['p80']:+.0f} days",
                f"on-time chance {100 * base['p_on_time']:.0f}% → {100 * tight['p_on_time']:.0f}%", delta_color="off")
    st.caption("Monte Carlo on the building's CPM network (Schedule page), with the pessimistic durations of electrical, mechanical and "
               "commissioning work widened by the tightness adjustment. The widening rule (up to +30% at index 100) is an assumption to calibrate "
               "against your own crew data.")


def _tariffs(d, ctx):
    howto("Landed cost = FOB in USD + freight + insurance + <b>duty stack</b> + broker + inland freight. US duty is charged on customs value "
          "(transaction value, excluding international freight). Stacking rule built in: the temporary Section 122 tariff does not apply to "
          "Section 232 goods. Every rate carries its legal basis, dates, source and whether it is verified. Edit the table to add or update rates.")
    tar = st.data_editor(d["tariffs"], num_rows="dynamic", width="stretch", key="tar_ed",
                         column_config={"source": st.column_config.LinkColumn(), "rate_pct": st.column_config.NumberColumn(format="%.1f")})
    scm = scm_model(ctx)
    items = scm["plan_items"]
    a, b, c = st.columns(3)
    asof = a.date_input("Duties as of", DATA_DATE, key="tar_asof")
    extra = b.slider("Scenario: extra tariff on all imports (%)", 0, 50, 0, 5)
    fxchg = c.slider("Scenario: exporter currencies move vs USD (%)", -20, 20, 0, 1, help="Negative = their currency falls")
    pt = st.slider("Share of the currency move passed into USD prices", 0.0, 1.0, 0.5, 0.05,
                   help="Studies of the 2018-19 US tariffs found US importers bore most of the tariff, so a low pass-through is the cautious default.")
    rows = []
    for it in items[items.origin != "USA"].itertuples():
        fob = it.qty_per_building * it.unit_cost_usd
        r0 = X.landed_cost(fob, it.item_id, it.origin, tar, asof)
        r1 = X.landed_cost(fob, it.item_id, it.origin, tar, asof, extra_tariff_pct=extra, exporter_fx_change_pct=fxchg, fx_pass_through=pt)
        rows.append({"item_id": it.item_id, "item": it.name, "origin": it.origin, "fob_per_building": fob, "duty_basis": r0["duty_basis"],
                     "duty_rate_%": r0["duty_rate_%"], "duty": r0["duty"], "landed": r0["landed_usd"], "landed_scenario": r1["landed_usd"],
                     "scenario_change": r1["landed_usd"] - r0["landed_usd"], "landed_vs_fob_%": 100 * (r0["landed_usd"] / fob - 1)})
    lc = pd.DataFrame(rows)
    k = st.columns(3)
    k[0].metric("Duty per building (imports)", money(lc.duty.sum()))
    k[1].metric("Landed cost per building (imports)", money(lc.landed.sum()))
    k[2].metric("Scenario change per building", money(lc.scenario_change.sum()), delta_color="inverse")
    st.dataframe(lc, hide_index=True, width="stretch",
                 column_config={"fob_per_building": DOLLAR, "duty": DOLLAR, "landed": DOLLAR, "landed_scenario": DOLLAR, "scenario_change": DOLLAR,
                                "duty_rate_%": NUM, "landed_vs_fob_%": NUM})
    st.caption("Net-export logic (Marthinsen, Ch. 17): a tariff lowers demand for the exporter's currency, which tends to fall and offset part of "
               "the tariff for the buyer. Model it with the two scenario sliders, but plan on full pass-through. Not customs or legal advice: "
               "confirm HTS classification and current rates with a licensed broker.")


def _fx(d, ctx):
    howto("Every open order in a foreign currency, by payment date. <b>Hedged</b>: the forward locks the dollar cost today. <b>Unhedged</b>: the "
          "95% band comes from each currency's own volatility over the last 3 years and grows with time to payment.")
    scm = scm_model(ctx)
    ex = X.fx_exposure(scm["d"]["open_pos"], d["monetary"], DATA_DATE)
    if not len(ex):
        st.info("No open foreign-currency orders.")
        return
    hr = st.slider("Share of exposure hedged with forwards", 0.0, 1.0, 0.0, 0.1, key="fx_hr")
    g = ex.groupby("currency").agg(usd_at_order=("usd_at_order", "sum"), usd_at_spot=("usd_at_spot", "sum"), usd_forward=("usd_hedged_forward", "sum"),
                                   low=("usd_unhedged_low", "sum"), high=("usd_unhedged_high", "sum"), vol=("annual_vol", "mean")).reset_index()
    g["band_unhedged"] = g.high - g.low
    g["band_after_hedge"] = g.band_unhedged * (1 - hr)
    g["fx_gain_loss_vs_order"] = g.usd_at_order - g.usd_at_spot
    k = st.columns(4)
    k[0].metric("Open foreign-currency orders (at spot)", money(g.usd_at_spot.sum()))
    k[1].metric("Gain (+) / loss (−) since order", money(g.fx_gain_loss_vs_order.sum()))
    k[2].metric("95% range if unhedged", money(g.band_unhedged.sum()))
    k[3].metric("95% range at this hedge ratio", money(g.band_after_hedge.sum()))
    st.dataframe(g, hide_index=True, width="stretch", column_config={c: DOLLAR for c in ["usd_at_order", "usd_at_spot", "usd_forward", "low", "high",
                                                                                          "band_unhedged", "band_after_hedge", "fx_gain_loss_vs_order"]}
                 | {"vol": st.column_config.NumberColumn("annual vol", format="%.3f")})
    ch = alt.Chart(g).mark_rule(strokeWidth=10, color=SERIES[0], opacity=0.6).encode(x=alt.X("low:Q", title="USD cost (95% band if unhedged)"), x2="high:Q",
                                                                                      y=alt.Y("currency:N", title=None))
    dot = alt.Chart(g).mark_point(shape="diamond", size=90, filled=True, color=STATUS["warning"]).encode(x="usd_forward:Q", y="currency:N",
                                                                                                         tooltip=["currency", alt.Tooltip("usd_forward:Q", format="$,.0f")])
    st.altair_chart((ch + dot).properties(height=alt.Step(30), title="Band if unhedged; diamond = locked forward cost"), width="stretch")
    with st.expander("Payment-level detail"):
        st.dataframe(ex, hide_index=True, width="stretch", column_config={c: DOLLAR for c in ["usd_at_order", "usd_at_spot", "usd_hedged_forward", "usd_hedged_after_shock",
                                                                                              "usd_unhedged_low", "usd_unhedged_high"]})


def _risk(d, ctx):
    howto("Ten drivers for every critical item and its country of origin. Score = likelihood (1-3) × speed (how fast it would hit, 1-3) × "
          "exposure weight (dollar tercile, 1-3), ×1.5 if the item is critical. Edit likelihoods to match your own judgement. Each high score "
          "comes with a standard response.")
    scm = scm_model(ctx)
    items = scm["plan_items"]
    op = scm["d"]["open_pos"].drop_duplicates(["po_id", "milestone"]).groupby("item_id").amount_usd_at_order.sum()
    rem = (scm["req"].groupby("item_id").gross.sum() * items.set_index("item_id").unit_cost_usd).fillna(0.0)
    exposure = rem.add(op, fill_value=0.0)
    seed = st.data_editor(d["risk_seed"].pivot(index="origin", columns="driver", values="likelihood")[X.DRIVERS], width="stretch", key="risk_ed",
                          column_config={c: st.column_config.NumberColumn(min_value=1, max_value=3, step=1) for c in X.DRIVERS})
    seed_long = seed.reset_index().melt(id_vars="origin", var_name="driver", value_name="likelihood")
    reg = X.risk_register(items, exposure, seed_long)
    hm = reg.groupby(["origin", "driver"]).score.max().reset_index()
    st.altair_chart(alt.Chart(hm).mark_rect(cornerRadius=3).encode(
        x=alt.X("driver:N", title=None, sort=X.DRIVERS, axis=alt.Axis(labelAngle=-30)), y=alt.Y("origin:N", title=None),
        color=alt.Color("score:Q", scale=alt.Scale(scheme="orangered"), legend=alt.Legend(title="max score")),
        tooltip=["origin", "driver", alt.Tooltip("score:Q", format=".1f")]).properties(height=alt.Step(26)), width="stretch")
    lvl = st.multiselect("Show ratings", ["High", "Medium", "Low"], ["High"], key="risk_lvl")
    st.dataframe(reg[reg.rating.isin(lvl)], hide_index=True, width="stretch", column_config={"exposure_usd": DOLLAR, "score": NUM})


def _ledger(d, ctx):
    howto("One shock, every channel, in dollars, only the part that lands before the program ends. <b>Fast</b> channels (FX, forwards, "
          "financing) land in full; <b>slow</b> ones (prices, wages) only partly. Assumptions are sliders, so the answer can be challenged.")
    c1, c2, c3 = st.columns(3)
    with c1:
        code = _country_pick(d, "led_c", "US")
    dr = c2.slider("Policy rate change (pp)", -2.0, 2.0, 0.5, 0.25, key="led_dr")
    ml = c3.slider("Months left in the program", 1, 36, _months_left(ctx), key="led_ml")
    a, b, c, e = st.columns(4)
    wc = a.number_input("Contractors' financed working capital ($)", 0.0, 1e9, 40e6, 5e6, help="Roughly 60 days of billing + retention")
    hedged = b.slider("Share of FX exposure hedged", 0.0, 1.0, 0.0, 0.1, key="led_h")
    elast = c.slider("Material price change per +1pp at full effect", -0.05, 0.0, -0.01, 0.005, format="%.3f")
    appr = e.slider("Currency move per +1pp (appreciation)", 0.0, 0.05, 0.02, 0.005, format="%.3f")
    scm = scm_model(ctx)
    items = scm["plan_items"].set_index("item_id")
    rem = scm["req"].groupby("item_id").gross.sum() * items.unit_cost_usd
    by_origin = rem.groupby(items.origin.map(X.ORIGIN_COUNTRY)).sum()
    inv = scm["bal"].value_usd.sum()
    t = X.impact_ledger(code, dr, d["monetary"], DATA_DATE, scm["d"]["open_pos"], inv, by_origin, ml, wc, price_elasticity=elast,
                        appreciation_per_pp=appr, hedged_share=hedged)
    total = t.usd.sum()
    st.metric("Net effect on the program inside the window (+ = costs more)", money(total))
    st.dataframe(t, hide_index=True, width="stretch", column_config={"usd": DOLLAR})
    st.altair_chart(alt.Chart(t).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("usd:Q", title="USD (+ costs more)"), y=alt.Y("channel:N", title=None, sort=None, axis=alt.Axis(labelLimit=360)),
        color=alt.Color("speed:N", scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="bottom", title="speed")),
        tooltip=["channel", "speed", alt.Tooltip("usd:Q", format="$,.0f"), "note"]).properties(height=alt.Step(30)), width="stretch")
    st.caption("Material price channel uses the value of material still to come from that country (from the material plan). A US move also "
               "moves every forward, because the US rate is in every forward formula.")


def _method():
    st.markdown("""
**Three-sector model** (Marthinsen, *Demystifying Global Macroeconomics*): describe the setting (where the economy sits on aggregate supply,
FX regime, capital mobility, central-bank credibility), identify the exogenous shock, then follow the chain through the real credit market,
the real goods and services market and the foreign exchange market. Direct effects outweigh indirect ones; never start from a price.

**Formulas used**
- Money: B = CC + RR + ER · M2 = CC + D + N · mm2 = (1 + c + n)/(c + rd + rn·n + e)
- Exact Fisher: real = (1 + i)/(1 + expected inflation) − 1
- Covered interest parity: forward (foreign per USD) = spot × ((1 + i_foreign)/(1 + i_USD))^t
- Balance of payments: CA + KA + NEO = FA + RRI · net exports ≈ (S − I) + (T − G)
- Reserve tests: months of import cover (rule of thumb ≥ 3) · Greenspan–Guidotti reserves ÷ short-term external debt (≥ 1)
- Intervention: Δbase = − FX sold × rate × (1 − sterilized share)

**Where the method changes**: in floor systems (Fed, ECB, BoJ) the rate is moved by paying more on reserves; the base need not move and
reserves are not the constraint, so total money follows loan demand. In corridor systems with scarce reserves (RBI liquidity operations,
reserve-ratio changes) the multiplier model applies. US M1 has included savings deposits since May 2020; India's headline broad money is M3.

**Assumptions to calibrate** (sliders): base drain per pp, loan-demand response, currency move per pp, price response and its lag,
labor widening rule. Where theory says the direction is uncertain, the app says so.

**Real data, same columns**: FRED (BOGMBASE, M1SL, M2SL, CURRCIR, WRESBAL, IORB, DFEDTARU, T5YIE, DEXINUS, DEXKOUS, DEXMXUS, DEXJPUS,
DEXCHUS, DEXUSEU), RBI DBIE (reserve money, M1, M3, weekly FX reserves), IMF BOP/IIP, BLS LAUS (metro unemployment), CES (construction jobs and
earnings), OEWS (electrician wages). Tariffs: CBP and the Federal Register, confirmed by a licensed customs broker.

**Not advice**: this is a planning aid, not investment, hedging, customs or legal advice.
""")
