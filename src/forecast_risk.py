"""
Probabilistic forecasting for one project (Business Analytics: Monte Carlo + sensitivity; Forecasting: backtests).

monte_carlo()   cost and finish as distributions (P10/P50/P80/P90), chance of finishing within budget / on time
tornado()       one-way sensitivity: how far each uncertain input moves forecast cost and finish
backtest()      pseudo-out-of-sample test of progress forecasts: MAE, MAPE and bias of two methods
All assumptions are explicit inputs with defaults, shown and editable in the app.
"""
import numpy as np
import pandas as pd

DEFAULTS = {
    "p_approve": 0.60,        # chance each pending change order is approved
    "negotiate_lo": 0.70,     # approved changes settle at 70%-100% of the amount asked
    "growth_mode": 1.0,       # unforeseen new changes, % of remaining work (triangular 0 / mode / max)
    "growth_max": 4.0,
    "material_share": 40.0,   # share of remaining work that is materials exposed to price moves
    "esc_mean": 1.0,          # material price change over the remaining duration, % (normal mean / sd)
    "esc_sd": 3.0,
    "pace_cv_floor": 0.15,    # minimum variability of the remaining duration
}


def pace_stats(prog_p):
    """Monthly verified-progress increments (points) over months with progress; mean and coefficient of variation."""
    s = prog_p.dropna(subset=["verified_pct"]).sort_values("period").verified_pct
    inc = s.diff().dropna()
    inc = inc[inc > 0.05].tail(6)
    if len(inc) < 2:
        return np.nan, np.nan
    return float(inc.mean()), float(inc.std(ddof=1) / inc.mean())


def _pending_cos(tables, project_id):
    co = tables["change_orders"].merge(tables["purchase_orders"][["po_id", "project_id"]], on="po_id", suffixes=("", "_po"))
    pid_col = "project_id" if "project_id" in co.columns else "project_id_po"
    return co[(co[pid_col] == project_id) & (co.status == "PENDING")].co_amount.astype(float).values


def monte_carlo(tables, row, prog_p, n=5000, seed=7, a=None):
    """row: one project's program_table row (committed, original_contract, approved_co, budget, verified_pct,
    planned_finish, forecast_finish, data_date, ld_rate, ld_cap)."""
    a = {**DEFAULTS, **(a or {})}
    rng = np.random.default_rng(seed)
    pend = _pending_cos(tables, row["project_id"])
    remaining = max(0.0, row["committed"] - (row["verified"] if pd.notna(row["verified"]) else 0.0))
    approve = rng.random((n, len(pend))) < a["p_approve"]
    settle = rng.uniform(a["negotiate_lo"], 1.0, (n, len(pend)))
    co_cost = (approve * settle * pend).sum(axis=1) if len(pend) else np.zeros(n)
    growth = rng.triangular(0, min(a["growth_mode"], a["growth_max"]), max(a["growth_max"], 1e-6), n) / 100 * remaining
    esc = rng.normal(a["esc_mean"], a["esc_sd"], n) / 100 * remaining * a["material_share"] / 100
    cost = row["committed"] + co_cost + growth + esc
    out = {"cost": cost, "pending_total": float(pend.sum()), "remaining": remaining}
    # finish: remaining duration to the base forecast, scaled by a lognormal pace factor
    dd, ff = row.get("data_date"), row.get("forecast_finish")
    if pd.notna(dd) and pd.notna(ff):
        base_days = max(1.0, (ff - dd).days)
        _, cv = pace_stats(prog_p)
        cv = max(a["pace_cv_floor"], cv if pd.notna(cv) else 0.3)
        sigma = np.sqrt(np.log(1 + cv ** 2))
        factor = rng.lognormal(-sigma ** 2 / 2, sigma, n)
        finish = dd + pd.to_timedelta(base_days * factor, unit="D")
        out["finish"] = finish
        if pd.notna(row.get("planned_finish")):
            late_w = np.clip((finish - row["planned_finish"]).days / 7, 0, None)
            out["ld"] = np.minimum(row["ld_rate"] * late_w, row["ld_cap"]) / 100 * row["committed"]
    return out


def summarize(mc, row):
    c = mc["cost"]
    s = {"cost_p10": np.percentile(c, 10), "cost_p50": np.percentile(c, 50), "cost_p80": np.percentile(c, 80),
         "cost_p90": np.percentile(c, 90), "cost_mean": c.mean()}
    if pd.notna(row.get("budget")):
        s["p_within_budget"] = float((c <= row["budget"]).mean())
        s["contingency_for_p80"] = s["cost_p80"] - row["budget"]
    if "finish" in mc:
        f = pd.Series(mc["finish"])
        s.update({f"finish_p{q}": f.quantile(q / 100) for q in (10, 50, 80, 90)})
        if pd.notna(row.get("planned_finish")):
            s["p_on_time"] = float((f <= row["planned_finish"]).mean())
    if "ld" in mc:
        s["ld_p50"], s["ld_p80"] = np.percentile(mc["ld"], 50), np.percentile(mc["ld"], 80)
    return s


def tornado(tables, row, prog_p, a=None):
    """Swing in median cost ($) and median finish (days) as each input moves from its low to its high value."""
    base = {**DEFAULTS, **(a or {})}
    _, cv = pace_stats(prog_p)
    cv = max(base["pace_cv_floor"], cv if pd.notna(cv) else 0.3)
    tests = [
        ("Pending change orders approved", "p_approve", 0.3, 0.9),
        ("Negotiated settlement level", "negotiate_lo", 0.6, 0.95),
        ("Unforeseen new changes (max %)", "growth_max", 1.0, 8.0),
        ("Material price change (mean %)", "esc_mean", -2.0, 5.0),
        ("Share of work that is materials", "material_share", 25.0, 55.0),
    ]
    med = lambda x: float(np.median(x))  # noqa: E731
    b = monte_carlo(tables, row, prog_p, n=3000, a=base)
    b_cost = med(b["cost"])
    rows = []
    for label, key, lo, hi in tests:
        vals = []
        for v in (lo, hi):
            r = monte_carlo(tables, row, prog_p, n=3000, a={**base, key: v})
            vals.append(med(r["cost"]) - b_cost)
        rows.append({"input": label, "low": lo, "high": hi, "cost_low": vals[0], "cost_high": vals[1],
                     "swing": abs(vals[1] - vals[0]), "measure": "cost"})
    # pace variability drives the finish date
    if "finish" in b:
        b_fin = pd.Series(b["finish"]).median()
        fins = []
        for f in (0.5, 2.0):
            r = monte_carlo(tables, row, prog_p, n=3000, a={**base, "pace_cv_floor": max(0.05, cv * f)})
            fins.append((pd.Series(r["finish"]).quantile(0.8) - b_fin).days)
        rows.append({"input": "Pace variability (P80 finish)", "low": cv * 0.5, "high": cv * 2, "cost_low": 0, "cost_high": 0,
                     "days_low": fins[0], "days_high": fins[1], "swing": abs(fins[1] - fins[0]), "measure": "finish"})
    return pd.DataFrame(rows), b_cost


def backtest(prog, horizons=(1, 2)):
    """Pseudo-out-of-sample test: at each past month, forecast verified % h months ahead, compare with what happened.
    Methods: 'Recent pace' (average of the last 3 monthly gains) and 'Plan-adjusted' (current gap to plan carried forward)."""
    rows = []
    for pid, g in prog.groupby("project_id"):
        g = g.dropna(subset=["verified_pct"]).sort_values("period").reset_index(drop=True)
        for k in range(3, len(g)):
            for h in horizons:
                if k + h >= len(g):
                    continue
                actual = g.verified_pct[k + h]
                if actual >= 99.5 or g.verified_pct[k] < 1:
                    continue
                gains = g.verified_pct.diff()[k - 2:k + 1]
                pace = min(100.0, g.verified_pct[k] + h * gains.mean())
                rows.append({"project_id": pid, "cut": g.period[k], "h": h, "method": "Recent pace", "forecast": pace, "actual": actual})
                if pd.notna(g.planned_pct[k]) and pd.notna(g.planned_pct[k + h]):
                    plan = min(100.0, g.verified_pct[k] + (g.planned_pct[k + h] - g.planned_pct[k]))
                    rows.append({"project_id": pid, "cut": g.period[k], "h": h, "method": "Plan-adjusted", "forecast": plan, "actual": actual})
    df = pd.DataFrame(rows)
    if not len(df):
        return df, pd.DataFrame()
    df["err"] = df.forecast - df.actual
    summ = df.groupby(["method", "h"]).apply(lambda d: pd.Series({
        "tests": len(d), "MAE (pts)": d.err.abs().mean(), "RMSE (pts)": np.sqrt((d.err ** 2).mean()),
        "MAPE (%)": 100 * (d.err.abs() / d.actual.clip(lower=1)).mean(), "Bias (pts)": d.err.mean()}),
        include_groups=False).reset_index()
    return df, summ
