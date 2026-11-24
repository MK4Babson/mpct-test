"""
Cost intelligence: is an overrun the market's fault or the contractor's, is a rate fair, and how much could prices cost us?

  F6  beta / alpha     CAPM-style regression of package cost growth on its trade's price index
  F7  should-cost      hedonic regression of past bid rates (log rate ~ item + quantity + region + labour + timing + index)
  F8  market benchmark contract rate escalated to today with the price index
  F9  Cost-at-Risk     analytical, historical and Monte Carlo VaR on the unbilled material exposure, plus stress scenarios

Guardrails (the Altreya lesson): results come with standard errors, intervals and the factors the model cannot see.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from build_extras import FAMILY, TRADE_BASKET, REGIONS

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "market"
INDEX_COLS = ["steel", "copper", "concrete", "labor", "equipment"]
TRADE_OF_PRIME = {"E": "Electrical", "M": "Mechanical", "C": "Civil"}
MATERIAL_SHARE = {"Electrical": 0.45, "Mechanical": 0.50, "Civil": 0.40}
FRED_SUGGEST = {
    "steel": ("WPU1017", "PPI: Steel mill products"),
    "copper": ("WPU10260314", "PPI: Copper wire and cable"),
    "concrete": ("PCU327320327320", "PPI: Ready-mix concrete manufacturing"),
    "labor": ("CES2000000003", "Average hourly earnings, construction"),
    "equipment": ("WPU117", "PPI: Electrical machinery and equipment"),
}


# ---------------------------------------------------------------- data
def load_indices(uploads=None):
    """Synthetic indices, with any uploaded FRED/BLS CSVs replacing their column. uploads: {column: DataFrame}."""
    idx = pd.read_csv(MARKET / "indices_synthetic.csv")
    sources = {c: "synthetic" for c in INDEX_COLS}
    for col, df in (uploads or {}).items():
        s = fred_series(df)
        if s is None or len(s) < 24:
            continue
        s = 100 * s / s.iloc[0]
        m = idx[["period"]].merge(s.rename(col).reset_index(), on="period", how="left")
        if m[col].notna().sum() >= 24:
            idx[col] = m[col].interpolate(limit_direction="both").round(2)
            sources[col] = "uploaded"
    return idx, sources


def fred_series(df):
    """Read a FRED download (observation_date/DATE + value) or a BLS-style period,value CSV into a monthly series."""
    cols = [c.lower() for c in df.columns]
    df = df.copy()
    df.columns = cols
    dcol = next((c for c in cols if c in ("observation_date", "date", "period", "month")), cols[0])
    vcol = next((c for c in cols if c != dcol), None)
    if vcol is None:
        return None
    d = pd.to_datetime(df[dcol], errors="coerce")
    v = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(v.values, index=d.dt.to_period("M").astype(str)).dropna()
    s = s[~s.index.duplicated()]
    s.index.name = "period"
    return s.sort_index()


def trade_index(idx):
    ix = idx.set_index("period")[INDEX_COLS]
    return pd.DataFrame({tr: sum(w * ix[k] for k, w in b.items()) for tr, b in TRADE_BASKET.items()})


def trade_returns(idx):
    r = idx.set_index("period")[INDEX_COLS].pct_change() * 100
    return pd.DataFrame({tr: sum(w * r[k] for k, w in b.items()) for tr, b in TRADE_BASKET.items()})


# ---------------------------------------------------------------- OLS helper
def ols(y, X, names):
    """Plain OLS with classical standard errors, t, p, R2, adj R2, AIC, BIC. X includes the constant."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    sigma2 = sse / (n - k)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst
    ll = -n / 2 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
    return {"coef": pd.DataFrame({"term": names, "estimate": beta, "std_error": se, "t": t, "p_value": p}),
            "beta": beta, "resid": resid, "sigma2": sigma2, "xtx_inv": xtx_inv, "n": n, "k": k, "sse": sse,
            "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / (n - k), "aic": 2 * k - 2 * ll, "bic": k * np.log(n) - 2 * ll}


# ---------------------------------------------------------------- F6 beta / alpha
def beta_alpha(hist=None):
    """Per trade: cost_growth = alpha + beta * index_change. Returns table + scatter data."""
    hist = hist if hist is not None else pd.read_csv(MARKET / "history_monthly.csv")
    rows = []
    for tr, d in hist.groupby("trade"):
        f = ols(d.cost_growth_pct, np.column_stack([np.ones(len(d)), d.index_change_pct]), ["alpha", "beta"])
        c = f["coef"].set_index("term")
        tcrit = stats.t.ppf(0.975, f["n"] - 2)
        rows.append({"trade": tr, "beta": c.estimate["beta"], "beta_lo": c.estimate["beta"] - tcrit * c.std_error["beta"],
                     "beta_hi": c.estimate["beta"] + tcrit * c.std_error["beta"], "beta_p": c.p_value["beta"],
                     "alpha_pct_month": c.estimate["alpha"], "alpha_p": c.p_value["alpha"], "r2": f["r2"], "months": f["n"],
                     "packages": d.package_id.nunique()})
    # alpha per contractor tier (controls for the market)
    tiers = []
    for (tr, tier), d in hist.groupby(["trade", "contractor_tier"]):
        f = ols(d.cost_growth_pct, np.column_stack([np.ones(len(d)), d.index_change_pct]), ["alpha", "beta"])
        c = f["coef"].set_index("term")
        tiers.append({"trade": tr, "tier": tier, "alpha_pct_month": c.estimate["alpha"], "alpha_se": c.std_error["alpha"],
                      "alpha_p": c.p_value["alpha"], "months": f["n"]})
    return pd.DataFrame(rows), pd.DataFrame(tiers), hist


def overrun_split(tables, idx, ba, data_period, award_offset=-1):
    """Split each project-trade package's cost growth (EAC vs original contract) into market-driven vs project-specific."""
    po, pol, co = tables["purchase_orders"], tables["po_lines"], tables["change_orders"]
    orig = pol[pol.line_source == "CONTRACT"].groupby("po_id").scheduled_value.sum()
    co_all = co[co.status.isin(["APPROVED", "PENDING"])].groupby("po_id").co_amount.sum()
    ti = trade_index(idx)
    betas = ba.set_index("trade").beta
    data_date = data_period if data_period in ti.index else idx.period.iloc[-1]
    prj = tables.get("projects", pd.DataFrame())
    start = prj.set_index("project_id").start if "start" in prj.columns else pd.Series(dtype=float)
    rows = []
    for r in po.itertuples():
        trade = TRADE_OF_PRIME.get(str(r.prime_id)[-1])
        if trade is None or r.po_id not in orig.index:
            continue
        s = int(start.get(r.project_id, 0)) if len(start) else 0
        award = str(pd.Period("2026-01", "M") + s + award_offset)
        if award not in ti.index or data_date not in ti.index:
            continue
        mkt_move = 100 * (ti.loc[data_date, trade] / ti.loc[award, trade] - 1)
        growth = 100 * co_all.get(r.po_id, 0) / orig[r.po_id]
        market = betas.get(trade, np.nan) * mkt_move
        rows.append({"project_id": r.project_id, "po_id": r.po_id, "prime_id": r.prime_id, "trade": trade, "award": award,
                     "original": orig[r.po_id], "growth_pct": growth, "index_move_pct": mkt_move, "market_pct": market,
                     "specific_pct": growth - market, "market_usd": orig[r.po_id] * market / 100,
                     "specific_usd": orig[r.po_id] * (growth - market) / 100})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- F7 / F8 should-cost
SPECS = {
    "A: item + price index": ["log_index"],
    "B: + quantity (economies of scale)": ["log_index", "log_qty_rel"],
    "C: + region, union labour, acceleration": ["log_index", "log_qty_rel", "union_labor", "accelerated"] + [f"reg_{r}" for r in list(REGIONS)[1:]],
    "D: + site difficulty (full)": ["log_index", "log_qty_rel", "union_labor", "accelerated", "site_difficulty"] + [f"reg_{r}" for r in list(REGIONS)[1:]],
}
TERM_LABEL = {"log_index": "Price index (log, elasticity)", "log_qty_rel": "Quantity (log, scale effect)",
              "union_labor": "Union labour", "accelerated": "Accelerated schedule", "site_difficulty": "Site difficulty (per level)",
              **{f"reg_{r}": f"Region: {r}" for r in REGIONS}}


def _design(df, idx, ref_qty):
    ix = idx.set_index("period")
    d = df.copy()
    d["index_name"] = d.item_code.map(lambda c: FAMILY[c][2])
    d["log_index"] = [np.log(ix.loc[p, n]) if p in ix.index else np.nan for p, n in zip(d.period, d.index_name)]
    d["log_qty_rel"] = np.log(d.quantity / d.item_code.map(ref_qty))
    for r in list(REGIONS)[1:]:
        d[f"reg_{r}"] = (d.region == r).astype(int)
    for c in sorted(FAMILY):
        d[f"fe_{c}"] = (d.item_code == c).astype(int)
    return d


def fit_should_cost(idx, hist=None):
    hist = hist if hist is not None else pd.read_csv(MARKET / "rate_history.csv")
    ref_qty = hist.groupby("item_code").quantity.median()
    d = _design(hist, idx, ref_qty).dropna(subset=["log_index"])
    y = np.log(d.unit_rate)
    fes = [f"fe_{c}" for c in sorted(FAMILY)]
    out = {}
    for name, terms in SPECS.items():
        cols = fes + terms
        f = ols(y, d[cols].values, cols)
        out[name] = {"fit": f, "cols": cols, "terms": terms}
    comp = pd.DataFrame([{"model": n, "terms": len(o["cols"]), "adj_r2": o["fit"]["adj_r2"], "aic": o["fit"]["aic"],
                          "bic": o["fit"]["bic"], "resid_sd": np.sqrt(o["fit"]["sigma2"])} for n, o in out.items()])
    best = comp.sort_values("bic").model.iloc[0]
    # joint F-test: do region dummies matter? (spec D vs D without region)
    full = out["D: + site difficulty (full)"]
    reg = [c for c in full["cols"] if c.startswith("reg_")]
    restr_cols = [c for c in full["cols"] if c not in reg]
    fr = ols(y, d[restr_cols].values, restr_cols)
    q = len(reg)
    F = ((fr["sse"] - full["fit"]["sse"]) / q) / (full["fit"]["sse"] / (full["fit"]["n"] - full["fit"]["k"]))
    pF = float(stats.f.sf(F, q, full["fit"]["n"] - full["fit"]["k"]))
    # variance inflation factors for the attribute terms
    vif = []
    T = d[full["terms"]].astype(float)
    for c in full["terms"]:
        others = [o for o in full["terms"] if o != c]
        r2 = ols(T[c], np.column_stack([np.ones(len(T)), T[others].values]), ["c"] + others)["r2"]
        vif.append({"term": TERM_LABEL.get(c, c), "vif": 1 / max(1e-9, 1 - r2)})
    # omitted-variable demonstration: spec C vs D coefficient on acceleration/union
    coefC = out["C: + region, union labour, acceleration"]["fit"]["coef"].set_index("term").estimate
    coefD = full["fit"]["coef"].set_index("term").estimate
    ovb = pd.DataFrame([{"term": TERM_LABEL[t], "without_site_difficulty": coefC.get(t), "with_site_difficulty": coefD.get(t)}
                        for t in ["union_labor", "accelerated", "log_qty_rel"]])
    return {"models": out, "comparison": comp, "best": best, "region_F": F, "region_p": pF, "vif": pd.DataFrame(vif),
            "ovb": ovb, "ref_qty": ref_qty, "data": d, "hist": hist}


def should_cost_lines(sc, tables, idx, attrs=None, model=None, level=0.90):
    """Predict should-cost (and prediction interval) for each current unit-price contract line."""
    attrs = {"region": "West Texas", "union_labor": 0, "accelerated": 1, "site_difficulty": 1, **(attrs or {})}
    model = model or sc["best"]
    m = sc["models"][model]
    pol = tables["po_lines"]
    po = tables["purchase_orders"].set_index("po_id")
    prj = tables.get("projects", pd.DataFrame())
    start = prj.set_index("project_id").start if "start" in prj.columns else pd.Series(dtype=float)
    cur = pol[(pol.line_source == "CONTRACT") & pol.item_code.isin(FAMILY)].copy()
    if not len(cur):
        return pd.DataFrame()
    cur["project_id"] = cur.po_id.map(po.project_id)
    cur["prime_id"] = cur.po_id.map(po.prime_id)
    cur["period"] = [str(pd.Period("2026-01", "M") + int(start.get(p, 0)) - 1) for p in cur.project_id]
    cur["quantity"] = cur.contract_qty
    for k, v in attrs.items():
        cur[k] = v
    d = _design(cur, idx, sc["ref_qty"])
    X = d[m["cols"]].values.astype(float)
    f = m["fit"]
    pred = X @ f["beta"]
    se_pred = np.sqrt(f["sigma2"] * (1 + np.einsum("ij,jk,ik->i", X, f["xtx_inv"], X)))
    tcrit = stats.t.ppf(0.5 + level / 2, f["n"] - f["k"])
    d["should_cost"] = np.exp(pred + f["sigma2"] / 2)
    d["pi_lo"], d["pi_hi"] = np.exp(pred - tcrit * se_pred), np.exp(pred + tcrit * se_pred)
    d["family"] = d.item_code.map(lambda c: FAMILY[c][0])
    d["gap_pct"] = 100 * (d.contract_unit_rate / d.should_cost - 1)
    d["z"] = (np.log(d.contract_unit_rate) - pred) / se_pred
    d["flag"] = np.where(d.contract_unit_rate > d.pi_hi, "Above range", np.where(d.contract_unit_rate < d.pi_lo, "Below range", "Within range"))
    d["excess_usd"] = np.where(d.flag == "Above range", (d.contract_unit_rate - d.pi_hi) * d.contract_qty, 0)
    # F8: escalate to today's index (what the same contract rate would be if bought now)
    ix = idx.set_index("period")
    today = idx.period.iloc[-1]
    ms = d.item_code.map(lambda c: FAMILY[c][3])
    ind = d.item_code.map(lambda c: FAMILY[c][2])
    esc = [ms_ * (ix.loc[today, n] / ix.loc[p, n] - 1) + (1 - ms_) * (ix.loc[today, "labor"] / ix.loc[p, "labor"] - 1)
           for ms_, n, p in zip(ms, ind, d.period)]
    d["escalation_to_today_pct"] = 100 * np.array(esc)
    d["rate_today"] = d.contract_unit_rate * (1 + np.array(esc))
    return d[["po_id", "project_id", "prime_id", "item_code", "family", "unit", "contract_qty", "contract_unit_rate", "should_cost",
              "pi_lo", "pi_hi", "gap_pct", "z", "flag", "excess_usd", "period", "escalation_to_today_pct", "rate_today"]]


# ---------------------------------------------------------------- F9 Cost-at-Risk
STRESS = {
    "Steel surge like 2021 (+60% steel over 12 months)": {"steel": 0.60},
    "Tariff shock (+25% steel, +15% copper)": {"steel": 0.25, "copper": 0.15},
    "Copper squeeze (+30% copper)": {"copper": 0.30},
    "Equipment backlog (+12% equipment)": {"equipment": 0.12},
    "Broad inflation (+8% everything)": {c: 0.08 for c in INDEX_COLS},
}


def exposure(pt, tables, pass_through=0.5):
    """Unbilled material exposure by project and index: remaining contract value x material share x basket weight x pass-through."""
    po = tables["purchase_orders"]
    pol = tables["po_lines"]
    inv = tables["invoices"]
    val = pol.groupby("po_id").scheduled_value.sum()
    billed = inv.groupby("po_id").gross_amount.sum()
    rows = []
    for r in po.itertuples():
        trade = TRADE_OF_PRIME.get(str(r.prime_id)[-1])
        if trade is None:
            continue
        remaining = max(0.0, val.get(r.po_id, 0) - billed.get(r.po_id, 0))
        for k, w in TRADE_BASKET[trade].items():
            rows.append({"project_id": r.project_id, "trade": trade, "index": k,
                         "exposure": remaining * MATERIAL_SHARE[trade] * w * pass_through, "remaining": remaining})
    return pd.DataFrame(rows)


def var_report(idx, exp, horizon=6, conf=0.95, n_sims=20000, seed=7):
    """Cost-at-Risk over `horizon` months: analytical (normal), historical (rolling windows), Monte Carlo (bootstrap)."""
    ix = idx.set_index("period")[INDEX_COLS]
    r = np.log(ix).diff().dropna()
    tot = exp.groupby("index").exposure.sum().reindex(INDEX_COLS).fillna(0)
    w = tot.values
    # portfolio monthly $ change series
    port = r.values @ w  # approx $ change per month (log-return x exposure)
    mu, sd = port.mean(), port.std(ddof=1)
    z = stats.norm.ppf(conf)
    analytical = max(0.0, mu * horizon + z * sd * np.sqrt(horizon))
    # historical: rolling h-month windows
    hroll = pd.Series(port).rolling(horizon).sum().dropna()
    historical = max(0.0, float(np.quantile(hroll, conf)))
    # Monte Carlo: block bootstrap of monthly joint returns (keeps cross-correlation)
    rng = np.random.default_rng(seed)
    R = r.values
    sims = np.empty(n_sims)
    for i in range(n_sims):
        st_ = rng.integers(0, len(R) - 3, size=int(np.ceil(horizon / 3)))
        block = np.vstack([R[s:s + 3] for s in st_])[:horizon]
        sims[i] = (np.exp(block.sum(0)) - 1) @ w
    mc = max(0.0, float(np.quantile(sims, conf)))
    cvar = float(sims[sims >= np.quantile(sims, conf)].mean())
    jb = stats.jarque_bera(port)
    stress = []
    for name, shock in STRESS.items():
        stress.append({"scenario": name, "cost_impact": float(sum(tot.get(k, 0) * v for k, v in shock.items()))})
    return {"analytical": analytical, "historical": historical, "monte_carlo": mc, "cvar": cvar, "sims": sims,
            "jb_stat": float(jb.statistic), "jb_p": float(jb.pvalue), "skew": float(stats.skew(port)),
            "kurtosis": float(stats.kurtosis(port)), "monthly_sd": sd, "exposure_by_index": tot, "stress": pd.DataFrame(stress),
            "monthly_change": port}
