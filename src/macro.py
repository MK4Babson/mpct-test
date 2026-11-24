"""
Global markets engine: the three macroeconomic markets (real credit, real goods & services, foreign exchange),
the balance of payments, FX reserves, local labor, tariffs, and a ledger of what each shock costs this program.

Method follows Marthinsen's three-sector model: describe the setting, identify the exogenous shock, follow the chain;
direct effects outweigh indirect ones; where theory says "uncertain", the app says so and shows what the data suggest.

Block A  Money        B = CC + RR + ER ; M2 = CC + D + N
                      mm2 = (1 + c + n) / (c + rd + rn*n + e),  mm1 = (1 + c) / (same),  c = CC/D, n = N/D, e = ER/D
                      a policy-rate change moves B (if the central bank drains or adds reserves) and the ratios
                      (estimated from history; textbook signs: c down, n up, e down when rates rise)
Block B  Credit       nominal rate = real + risk + tax + maturity premiums + expected inflation;
                      exact Fisher r = (1+i)/(1+pi_e) - 1; pass-through of the policy rate to lending rates (regression)
Block C  Goods        rate up -> C, I down -> AD down -> prices, real GDP down, spread over a lag (default peak ~12 months);
                      only the share of the lag inside the project window affects the project
Block D  FX           forward = spot * (1 + i_foreign) / (1 + i_usd) ^ t  (covered interest parity, quoted per USD);
                      exposure of open purchase orders by payment date; unhedged 95% band from historical volatility
Block E  BOP          CA + KA + NEO = FA + RRI check; reserve cover (months of imports), Greenspan-Guidotti
                      (reserves / short-term external debt), intervention -> monetary base, early-warning score
Labor                 local craft-labor tightness index -> wider schedule ranges on labor-heavy tasks, wage allowance
Trade                 landed cost = FOB(USD) + freight + insurance + duty stack + broker + inland; stacking rules
Risk                  ten-driver register for each critical item x country of origin
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "macro"
NAMES = ["countries", "monetary", "bop", "labor_metro", "competing_projects", "trade_supply", "tariffs", "risk_seed"]
CURRENCY_COUNTRY = {"USD": "US", "INR": "IN", "KRW": "KR", "MXN": "MX", "EUR": "EA", "JPY": "JP", "CNY": "CN"}
ORIGIN_COUNTRY = {"USA": "US", "India": "IN", "Korea": "KR", "Mexico": "MX", "Germany": "EA", "Japan": "JP", "China": "CN"}
DRIVERS = ["Labor & demographics", "Energy & raw materials", "Export policy (origin country)", "Import policy (US tariffs)",
           "Currency & macro", "Logistics chokepoints", "Natural hazards & climate", "Geopolitics & sanctions",
           "Concentration (single source)", "Quality & compliance"]
DRIVER_SPEED = dict(zip(DRIVERS, [1, 2, 3, 3, 3, 2, 3, 3, 2, 2]))   # 3 = hits within weeks, 1 = over many months
RESPONSE = {
    "Labor & demographics": "Ask the supplier for its capacity plan and shift pattern; qualify a second factory.",
    "Energy & raw materials": "Lock the price of the key input (index clause) or buy ahead of known shortages.",
    "Export policy (origin country)": "Contract clause for export-duty pass-through; keep a domestic alternative quoted.",
    "Import policy (US tariffs)": "Tariff table with dates; price the tariff-free alternative; consider bonded storage / FTZ timing.",
    "Currency & macro": "Hedge payment milestones with forwards; or price in USD with a capped FX clause.",
    "Logistics chokepoints": "Add route buffer days to the need date; book space early; alternative port of entry.",
    "Natural hazards & climate": "Weather and hazard buffer in the schedule; insure goods in transit.",
    "Geopolitics & sanctions": "Screen sub-suppliers; dual-source; avoid single-country exposure for critical items.",
    "Concentration (single source)": "Map tier-2 suppliers; qualify a second source; hold strategic buffer.",
    "Quality & compliance": "Factory acceptance test, mill certificates and standards check before shipment.",
}


def load(folder=FOLDER):
    d = {n: pd.read_csv(folder / f"{n}.csv") for n in NAMES if (folder / f"{n}.csv").exists()}
    d["monetary"]["month"] = pd.PeriodIndex(d["monetary"]["month"], freq="M")
    d["labor_metro"]["month"] = pd.PeriodIndex(d["labor_metro"]["month"], freq="M")
    d["tariffs"] = d["tariffs"].fillna("")
    return d


def latest(mon, country, month=None):
    x = mon[mon.country == country].sort_values("month")
    if month is not None:
        x = x[x.month <= pd.Period(month, "M")]
    return x.iloc[-1]


# ---------------------------------------------------------------- Block A: money
def aggregates(B, c, n, e, rd, rn):
    den = c + rd + rn * n + e
    D = B / den
    return {"monetary_base": B, "currency": c * D, "checkable_deposits": D, "near_money": n * D, "excess_reserves": e * D,
            "required_reserves": rd * D + rn * n * D, "M1": (1 + c) * D, "broad_money": (1 + c + n) * D, "mm1": (1 + c) / den,
            "mm2": (1 + c + n) / den, "c": c, "n": n, "e": e, "rd": rd, "rn": rn}


def ratios(row, ctry):
    D = row.checkable_deposits
    return {"c": row.currency / D, "n": row.near_money / D, "e": row.excess_reserves / D, "rd": float(ctry.rr_d), "rn": float(ctry.rr_n)}


PRIORS = {"c": -0.035, "n": 0.030, "e": -0.060}   # log-change per +1 percentage point (textbook signs, used if data disagree)


def sensitivities(mon, country, horizon=6):
    """Log-change of each preferred asset ratio per +1pp policy rate, from 6-month changes (OLS through the origin)."""
    x = mon[mon.country == country].sort_values("month")
    D = x.checkable_deposits
    lr = pd.DataFrame({"c": np.log(x.currency / D), "n": np.log(x.near_money / D), "e": np.log(x.excess_reserves / D)})
    dr = x.policy_rate.diff(horizon)
    out = {}
    for k in ["c", "n", "e"]:
        y = lr[k].diff(horizon)
        ok = dr.notna() & y.notna() & (dr.abs() > 0.05)
        if ok.sum() >= 12:
            b = float((dr[ok] * y[ok]).sum() / (dr[ok] ** 2).sum())
            good = np.sign(b) == np.sign(PRIORS[k])
            out[k] = (b if good else PRIORS[k], "estimated from history" if good else "textbook prior (data sign disagreed)")
        else:
            out[k] = (PRIORS[k], "textbook prior (too little rate movement)")
    return out


def money_shock(row, ctry, sens, d_rate, mode, base_drain_per_pp=0.06, d_rr=0.0, loan_semi_elasticity=-0.02, pass_through_=0.65):
    """Before/after of every monetary aggregate for a policy-rate change of d_rate percentage points.

    mode
      'drain'          scarce-reserves textbook case: open-market sales / liquidity absorption shrink the base by
                       base_drain_per_pp per pp; the public and banks re-balance their preferred asset ratios.
                       Broad money = mm2 x B (direct base effect + indirect multiplier effect).
      'reserve_ratio'  the reserve requirement changes by d_rr; base unchanged; the multiplier moves.
      'rate_only'      floor / ample-reserves system (Fed, ECB, BoJ): the rate paid on reserves moves, the base does not,
                       and reserves are not the constraint. Total money follows the credit channel (fewer loans ->
                       fewer deposits: loan_semi_elasticity per pp of lending-rate change); the ratios only change the
                       mix (cash -> time deposits). Banks do not run down reserves because reserves now pay more too.
    """
    r0 = ratios(row, ctry)
    before = aggregates(row.monetary_base, **r0)
    r1 = {k: r0[k] * np.exp(sens[k][0] * d_rate) for k in ("c", "n", "e")}
    r1.update(rd=r0["rd"] + (d_rr if mode == "reserve_ratio" else 0.0), rn=r0["rn"] + (d_rr if mode == "reserve_ratio" else 0.0))
    if mode == "rate_only":
        m2 = before["broad_money"] * (1 + loan_semi_elasticity * pass_through_ * d_rate)
        D = m2 / (1 + r1["c"] + r1["n"])
        cc, nn = r1["c"] * D, r1["n"] * D
        rr = r1["rd"] * D + r1["rn"] * nn
        B = row.monetary_base
        er = B - cc - rr
        after = {"monetary_base": B, "currency": cc, "checkable_deposits": D, "near_money": nn, "excess_reserves": er, "required_reserves": rr,
                 "M1": cc + D, "broad_money": m2, "mm1": (cc + D) / B, "mm2": m2 / B, "c": r1["c"], "n": r1["n"], "e": er / D}
    else:
        dB = -base_drain_per_pp * d_rate if mode == "drain" else 0.0
        after = aggregates(row.monetary_base * (1 + dB), **r1)
    keys = ["monetary_base", "currency", "checkable_deposits", "near_money", "excess_reserves", "M1", "broad_money", "mm1", "mm2", "c", "n", "e"]
    t = pd.DataFrame({"before": [before[k] for k in keys], "after": [after[k] for k in keys]}, index=keys)
    t["change_%"] = 100 * (t.after / t.before - 1)
    direct = np.log(after["monetary_base"] / before["monetary_base"])
    indirect = np.log(after["mm2"] / before["mm2"])
    lend0 = before["checkable_deposits"] + before["near_money"] - before["required_reserves"] - before["excess_reserves"]
    lend1 = after["checkable_deposits"] + after["near_money"] - after["required_reserves"] - after["excess_reserves"]
    return t, {"direct_base_%": 100 * direct, "indirect_multiplier_%": 100 * indirect, "lending_capacity_%": 100 * (lend1 / lend0 - 1),
               "mode": mode}


# ---------------------------------------------------------------- Block B: credit
def fisher_real(i, pi):
    return 100 * ((1 + i / 100) / (1 + pi / 100) - 1)


def pass_through(mon, country, horizon=6):
    x = mon[mon.country == country].sort_values("month")
    dp, dl = x.policy_rate.diff(horizon), x.lending_rate.diff(horizon)
    ok = dp.notna() & dl.notna() & (dp.abs() > 0.05)
    if ok.sum() < 12:
        return 0.6, "assumed (too little rate movement)"
    b = float((dp[ok] * dl[ok]).sum() / (dp[ok] ** 2).sum())
    return float(np.clip(b, 0.0, 1.2)), f"estimated from {int(ok.sum())} six-month changes"


# ---------------------------------------------------------------- Block C: goods market lag
def lag_weights(peak=12, length=24):
    m = np.arange(1, length + 1)
    w = np.where(m <= peak, m, np.maximum(length + 1 - m, 0) * peak / max(1, length - peak))
    return pd.Series(w / w.sum(), index=m)


def share_in_window(months_left, peak=12, length=24):
    w = lag_weights(peak, length)
    return float(w[w.index <= months_left].sum())


# ---------------------------------------------------------------- Block D: FX
def forward(spot, i_foreign, i_usd, years):
    """Forward rate quoted as foreign currency per USD (covered interest parity)."""
    return spot * ((1 + i_foreign / 100) / (1 + i_usd / 100)) ** years


def fx_vol(mon, country, months=36):
    x = mon[mon.country == country].sort_values("month").tail(months + 1)
    r = np.log(x.fx_per_usd).diff().dropna()
    return float(r.std(ddof=1) * np.sqrt(12)) if len(r) > 3 else 0.0


def fx_exposure(open_pos, mon, data_date, fx_at_order=None, rate_shock=None, appreciation_per_pp=0.02):
    """Open foreign-currency payments: USD cost at spot, at the forward, and a 95% band if unhedged.

    rate_shock: (country, d_rate_pp) applied to the foreign or US policy rate; moves forwards the same day and
    (by assumption appreciation_per_pp) the spot rate."""
    rows = []
    us = latest(mon, "US", data_date)
    for r in open_pos[open_pos.currency != "USD"].itertuples():
        cc = CURRENCY_COUNTRY.get(r.currency)
        if cc is None:
            continue
        x = latest(mon, cc, data_date)
        t = max(0.0, (pd.Timestamp(r.pay_date) - pd.Timestamp(data_date)).days / 365)
        i_f, i_u, spot = x.policy_rate, us.policy_rate, x.fx_per_usd
        f0 = forward(spot, i_f, i_u, t)
        spot1, i_f1, i_u1 = spot, i_f, i_u
        if rate_shock:
            sc, d = rate_shock
            if sc == cc:
                i_f1 = i_f + d
                spot1 = spot / (1 + appreciation_per_pp * d)          # foreign currency appreciates: fewer units per USD
            elif sc == "US":
                i_u1 = i_u + d
                spot1 = spot * (1 + appreciation_per_pp * d)          # dollar appreciates
        f1 = forward(spot1, i_f1, i_u1, t)
        vol = fx_vol(mon, cc)
        band = 1.645 * vol * np.sqrt(max(t, 1 / 52))
        usd_spot = r.amount_fx / spot
        rows.append({"po_id": r.po_id, "item_id": r.item_id, "milestone": r.milestone, "currency": r.currency, "pay_date": r.pay_date,
                     "amount_fx": r.amount_fx, "years": t, "usd_at_order": r.amount_usd_at_order, "usd_at_spot": usd_spot,
                     "usd_hedged_forward": r.amount_fx / f0, "usd_hedged_after_shock": r.amount_fx / f1,
                     "usd_unhedged_low": usd_spot / np.exp(band), "usd_unhedged_high": usd_spot * np.exp(band), "annual_vol": vol})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Block E: balance of payments and reserves
def bop_check(bop):
    b = bop.copy()
    b["lhs_CA_KA_NEO"] = b.current_account + b.capital_account + b.net_errors_omissions
    b["rhs_FA_RRI"] = b.financial_account + b.reserves_related_items
    b["difference"] = b.lhs_CA_KA_NEO - b.rhs_FA_RRI
    b["identity_holds"] = b.difference.abs() < 0.5
    b["net_lender"] = np.where(b.financial_account > 0, "net lender", "net borrower")
    b["central_bank"] = np.where(b.reserves_related_items > 0.5, "bought FX (adds to base)",
                                 np.where(b.reserves_related_items < -0.5, "sold FX (shrinks base)", "little intervention"))
    return b


def early_warning(bop, mon, countries, data_date):
    us_inf = latest(mon, "US", data_date).cpi_yoy
    rows = []
    for c in countries.itertuples():
        b = bop[bop.country == c.code].sort_values("year").iloc[-1]
        m = latest(mon, c.code, data_date)
        hist = mon[(mon.country == c.code)].sort_values("month")
        reer_avg = hist.tail(120).reer.mean()
        res_trend = hist.fx_reserves_usd_bn.iloc[-1] / hist.fx_reserves_usd_bn.iloc[-13] - 1
        ca_gdp = 100 * b.current_account / b.gdp
        cover = m.fx_reserves_usd_bn / (b.goods_imports / 12)
        gg = m.fx_reserves_usd_bn / b.short_term_external_debt
        pts, why = 0, []
        if ca_gdp < -5:
            pts += 2; why.append(f"current account {ca_gdp:.1f}% of GDP")
        elif ca_gdp < -3:
            pts += 1; why.append(f"current account {ca_gdp:.1f}% of GDP")
        if cover < 3:
            pts += 2; why.append(f"reserves cover {cover:.1f} months of imports")
        elif cover < 6:
            pts += 1; why.append(f"reserves cover {cover:.1f} months of imports")
        if gg < 1:
            pts += 2; why.append(f"reserves below short-term external debt ({gg:.2f}x)")
        if m.cpi_yoy - us_inf > 3:
            pts += 1; why.append(f"inflation {m.cpi_yoy - us_inf:+.1f}pp vs US")
        if m.reer > 1.10 * reer_avg:
            pts += 1; why.append("REER >10% above its 10-year average (overvalued?)")
        if "managed" in c.fx_regime and res_trend < -0.05:
            pts += 2; why.append(f"managed currency with reserves down {100 * res_trend:.0f}% in a year")
        if c.code == "US":
            pts, why = 0, ["issuer of the main reserve currency: reserve tests do not apply"]
        elif c.code in ("EA", "JP"):
            pts = sum(1 for w in why if "inflation" in w or "REER" in w)
            why = [w for w in why if "inflation" in w or "REER" in w] + ["reserve-currency issuer with a floating rate: reserve-cover tests are weak signals here"]
        rows.append({"country": c.code, "name": c.name, "ca_%gdp": ca_gdp, "reserves_usd_bn": m.fx_reserves_usd_bn, "import_cover_months": cover,
                     "greenspan_guidotti": gg, "inflation_gap_pp": m.cpi_yoy - us_inf, "reer_vs_10y_avg_%": 100 * (m.reer / reer_avg - 1),
                     "fx_regime": c.fx_regime, "score": pts, "level": "High" if pts >= 4 else "Medium" if pts >= 2 else "Low",
                     "drivers": "; ".join(why) or "no warning signs"})
    return pd.DataFrame(rows)


def intervention_to_base(fx_sold_usd_bn, fx_per_usd, sterilized_share):
    """Selling FX reserves to support the currency withdraws domestic currency: base falls unless offset (sterilized)."""
    return -fx_sold_usd_bn * fx_per_usd * (1 - sterilized_share)


# ---------------------------------------------------------------- labor
def labor_tightness(lab, competing, supply, data_date, months_ahead=12):
    end = pd.Period(data_date, "M") + months_ahead
    now = pd.Period(data_date, "M")
    rows = []
    for metro, g in lab.groupby("metro"):
        g = g.sort_values("month")
        g = g[g.month <= now]
        last, yr = g.iloc[-1], g.iloc[-13] if len(g) > 12 else g.iloc[0]
        u_pct = float((g.unemployment > last.unemployment).mean())                      # share of history with higher unemployment
        jobs_g = last.construction_jobs / yr.construction_jobs - 1
        wage_g = last.electrician_hourly / yr.electrician_hourly - 1
        cp = competing[competing.metro == metro]
        act = cp[(pd.PeriodIndex(cp.start, freq="M") <= end) & (pd.PeriodIndex(cp.finish, freq="M") >= now)]
        avail = float(supply.set_index("metro").electricians_available.get(metro, np.nan))
        comp_ratio = act.electricians_peak.sum() / avail if avail == avail and avail > 0 else 0.0
        idx = 100 * np.mean([u_pct, np.clip(jobs_g / 0.08, 0, 1), np.clip(wage_g / 0.08, 0, 1), np.clip(comp_ratio, 0, 1)])
        rows.append({"metro": metro, "unemployment": last.unemployment, "unemployment_lower_than_%_of_history": 100 * u_pct,
                     "construction_jobs_yoy_%": 100 * jobs_g, "electrician_wage_yoy_%": 100 * wage_g, "electrician_hourly": last.electrician_hourly,
                     "competing_projects": len(act), "competing_electricians_peak": int(act.electricians_peak.sum()), "electricians_available": avail,
                     "competing_share_of_supply_%": 100 * comp_ratio, "tightness_index": idx,
                     "level": "High" if idx >= 65 else "Medium" if idx >= 40 else "Low"})
    return pd.DataFrame(rows)


def labor_adjustments(index, wage_yoy_pct, months_left):
    """Turn tightness into planning numbers: how much to widen pessimistic durations on labor-heavy tasks, and a wage allowance."""
    widen = 0.30 * index / 100                                   # up to +30% on pessimistic duration at index 100 (assumption)
    premium = 0.04 * max(0.0, index - 50) / 50                   # premium time / per diem up to +4% when very tight (assumption)
    allowance = (1 + wage_yoy_pct / 100) ** (months_left / 12) - 1 + premium
    return {"pessimistic_widening": widen, "wage_allowance": allowance}


# ---------------------------------------------------------------- tariffs and landed cost
def duty_lines(tariffs, item_id, origin, as_of):
    if origin == "USA":
        return pd.DataFrame(columns=tariffs.columns)
    t = tariffs.copy()
    m = ((t.item_id == item_id) | (t.item_id == "*")) & ((t.origin == origin) | (t.origin == "*non-US*"))
    t = t[m]
    asof = pd.Timestamp(as_of)
    t = t[(pd.to_datetime(t.effective_from) <= asof) & ((t.effective_to == "") | (pd.to_datetime(t.effective_to.replace("", None)) >= asof))]
    if t.legal_basis.str.contains("232").any():                # Section 122 does not apply to Section 232 goods
        t = t[~t.legal_basis.str.contains("122")]
    return t


def landed_cost(fob_usd, item_id, origin, tariffs, as_of, freight_pct=0.06, insurance_pct=0.005, broker_usd=1500.0, inland_pct=0.02,
                extra_tariff_pct=0.0, exporter_fx_change_pct=0.0, fx_pass_through=1.0):
    """Landed cost in USD. US duties are charged on customs value (transaction value, excluding international freight).
    exporter_fx_change_pct < 0 means the exporter's currency fell; with pass-through the USD price falls."""
    fob = fob_usd * (1 + fx_pass_through * exporter_fx_change_pct / 100)
    lines = duty_lines(tariffs, item_id, origin, as_of)
    rate = float(lines.rate_pct.sum()) + (extra_tariff_pct if origin != "USA" else 0.0)
    duty = fob * rate / 100
    freight, ins, inland = fob * freight_pct, fob * insurance_pct, fob * inland_pct
    total = fob + freight + ins + duty + (broker_usd if origin != "USA" else 0.0) + inland
    return {"fob_usd": fob, "freight": freight, "insurance": ins, "duty_rate_%": rate, "duty": duty, "broker": broker_usd if origin != "USA" else 0.0,
            "inland": inland, "landed_usd": total, "duty_basis": ", ".join(lines.legal_basis) or "none (domestic or MFN 0)"}


# ---------------------------------------------------------------- risk register
def risk_register(items, exposure_usd, seed, critical_ids=()):
    rows = []
    ex = exposure_usd.to_dict()
    q = np.quantile(list(ex.values()), [1 / 3, 2 / 3]) if ex else [0, 0]
    sd = seed.set_index(["origin", "driver"]).likelihood
    for it in items.itertuples():
        e = ex.get(it.item_id, 0.0)
        w = 1 if e <= q[0] else 2 if e <= q[1] else 3
        for dvr in DRIVERS:
            lk = int(sd.get((it.origin, dvr), 1))
            crit = it.item_id in critical_ids or bool(it.critical)
            score = lk * DRIVER_SPEED[dvr] * w * (1.5 if crit else 1.0)
            rows.append({"item_id": it.item_id, "item": it.name, "origin": it.origin, "driver": dvr, "likelihood": lk, "speed": DRIVER_SPEED[dvr],
                         "exposure_usd": e, "exposure_weight": w, "critical": crit, "score": score, "response": RESPONSE[dvr]})
    r = pd.DataFrame(rows)
    r["rating"] = np.where(r.score >= 18, "High", np.where(r.score >= 9, "Medium", "Low"))
    return r.sort_values("score", ascending=False)


# ---------------------------------------------------------------- project impact ledger
def impact_ledger(country, d_rate, mon, data_date, open_pos, inventory_value, material_by_origin, months_left, wc_usd,
                  capital_rate=0.08, price_elasticity=-0.01, lag_peak=12, appreciation_per_pp=0.02, hedged_share=0.0):
    """Dollar effect on the program of a policy-rate change of d_rate pp in `country`, inside the project window.

    material_by_origin: USD value of material still to be bought/delivered, by origin country code.
    price_elasticity: change in that country's construction-input prices per +1pp at full effect (assumption)."""
    rows = []
    fx0 = fx_exposure(open_pos, mon, data_date)
    fx1 = fx_exposure(open_pos, mon, data_date, rate_shock=(country, d_rate), appreciation_per_pp=appreciation_per_pp)
    if len(fx0):
        aff = fx0.currency.map(CURRENCY_COUNTRY).eq(country) | (country == "US")
        hedge = float((fx1.usd_hedged_after_shock - fx0.usd_hedged_forward)[aff].sum())
        spot_new = fx1.usd_hedged_after_shock / fx0.usd_hedged_forward * fx0.usd_at_spot   # approx: spot moves with forward
        unhedged = float(((spot_new - fx0.usd_at_spot) * (1 - hedged_share))[aff].sum())
        rows.append({"channel": "Forward hedge cost on open foreign-currency orders", "speed": "Same day", "usd": hedge * hedged_share,
                     "note": f"{hedged_share:.0%} of exposure hedged; forwards move the day rates move"})
        rows.append({"channel": "Spot FX on unhedged payments", "speed": "Days to weeks", "usd": unhedged,
                     "note": f"assumes {appreciation_per_pp:.0%} appreciation per +1pp (assumption, wide uncertainty)"})
    if country == "US":
        pt, _ = pass_through(mon, "US")
        rows.append({"channel": "Contractors' working-capital financing (passed into bids and change orders)", "speed": "1-3 months",
                     "usd": wc_usd * d_rate / 100 * pt * months_left / 12, "note": f"pass-through {pt:.2f}, ${wc_usd / 1e6:.0f}M financed"})
        rows.append({"channel": "Owner's carrying cost on stock held", "speed": "1-3 months",
                     "usd": inventory_value * d_rate / 100 * months_left / 12, "note": "inventory value x rate change x time"})
    share = share_in_window(months_left, lag_peak)
    mat = float(material_by_origin.get(country, 0.0))
    rows.append({"channel": "Material prices (goods market, with lag)", "speed": "6-24 months",
                 "usd": mat * price_elasticity * d_rate * share,
                 "note": f"{share:.0%} of the lagged effect lands within {months_left} months; elasticity {price_elasticity:+.1%} per pp (assumption)"})
    t = pd.DataFrame(rows)
    return t
