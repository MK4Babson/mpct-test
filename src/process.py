"""
Process and planning analytics.

  F16  seasonal cash forecast   SARIMA vs Holt-Winters vs seasonal naive on the owner's monthly capital spend,
                                backtested; plus a bottom-up cash calendar for this program from its remaining plan
  F17  learning curve           log-log regression of cost per MW on building sequence (past buildings), applied to this program
  F18  invoice cycle time       stage durations from the approval log, Little's Law, bottleneck
  F26  embodied carbon          quantities x carbon factors (+ a spend-based factor for equipment)
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "market"


# ---------------------------------------------------------------- F16
def _fit_forecast(y, method, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "SARIMA(1,1,1)(0,1,1)12":
            m = SARIMAX(np.log(y), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
            f = m.get_forecast(h)
            mean, ci = np.exp(f.predicted_mean), np.exp(f.conf_int(alpha=0.2))
            return np.asarray(mean), np.asarray(ci)[:, 0], np.asarray(ci)[:, 1], m.aic
        if method == "Holt-Winters (multiplicative)":
            m = ExponentialSmoothing(y, trend="add", seasonal="mul", seasonal_periods=12).fit()
            mean = np.asarray(m.forecast(h))
            sd = np.std(m.resid[-36:], ddof=1)
            z = stats.norm.ppf(0.9)
            spread = z * sd * np.sqrt(np.arange(1, h + 1))
            return mean, mean - spread, mean + spread, m.aic
        # seasonal naive
        mean = np.asarray([y[-12 + (i % 12)] for i in range(h)])
        d = y[12:] - y[:-12]
        sd = np.std(d, ddof=1)
        z = stats.norm.ppf(0.9)
        spread = z * sd * np.sqrt(1 + np.arange(h) // 12)
        return mean, mean - spread, mean + spread, np.nan


METHODS = ["SARIMA(1,1,1)(0,1,1)12", "Holt-Winters (multiplicative)", "Seasonal naive"]


def cash_forecast(hist=None, h=6, backtest_from=24):
    hist = hist if hist is not None else pd.read_csv(MARKET / "portfolio_cash_history.csv")
    y = hist.capital_spend.values.astype(float)
    periods = pd.PeriodIndex(hist.period, freq="M")
    # rolling-origin backtest, 3-month-ahead
    bt = []
    for origin in range(len(y) - backtest_from, len(y) - 3 + 1, 3):
        for m in METHODS:
            mean, *_ = _fit_forecast(y[:origin], m, 3)
            act = y[origin:origin + 3]
            for k, (f_, a_) in enumerate(zip(mean, act)):
                bt.append({"method": m, "origin": str(periods[origin]), "h": k + 1, "forecast": f_, "actual": a_})
    bt = pd.DataFrame(bt)
    bt["err"] = bt.forecast - bt.actual
    acc = bt.groupby("method").apply(lambda d: pd.Series({
        "MAE ($M)": d.err.abs().mean() / 1e6, "RMSE ($M)": np.sqrt((d.err ** 2).mean()) / 1e6,
        "MAPE (%)": 100 * (d.err.abs() / d.actual).mean(), "Bias ($M)": d.err.mean() / 1e6}), include_groups=False).reset_index()
    best = acc.sort_values("MAPE (%)").method.iloc[0]
    fut = pd.period_range(periods[-1] + 1, periods=h, freq="M")
    fc = []
    for m in METHODS:
        mean, lo, hi, aic = _fit_forecast(y, m, h)
        fc.append(pd.DataFrame({"method": m, "period": fut.astype(str), "forecast": mean, "lo80": lo, "hi80": hi}))
    # seasonal profile: average ratio of each month to its 12-month centred average
    s = pd.Series(y, index=periods)
    trend = s.rolling(12, center=True).mean()
    ratio = (s / trend).dropna()
    season = ratio.groupby(ratio.index.month).mean()
    return {"history": hist, "backtest": bt, "accuracy": acc, "best": best, "forecast": pd.concat(fc), "season": season}


def program_cash_calendar(controls, pt, tables, last=None, pay_lag_days=30):
    """Bottom-up: remaining planned progress x committed value -> billing -> payment one term later, by month."""
    bp = controls.get("baseline_plan")
    if bp is None or pt is None or not len(pt):
        return pd.DataFrame()
    bp = bp.copy()
    bp["period"] = bp.period.astype(str)
    rows = []
    for r in pt.itertuples():
        b = bp[bp.project_id == r.project_id].sort_values("period")
        if not len(b):
            continue
        b["inc"] = b.planned_pct_cum.diff().fillna(b.planned_pct_cum)
        b = b[b.period > (last or b.period.min())]
        for x in b.itertuples():
            rows.append({"project_id": r.project_id, "name": r.name, "period": x.period, "billing": r.committed * x.inc / 100})
    cal = pd.DataFrame(rows)
    if not len(cal):
        return cal
    shift = int(round(pay_lag_days / 30))
    cal["pay_period"] = (pd.PeriodIndex(cal.period, freq="M") + shift).astype(str)
    retain = 0.10
    cal["cash_out"] = cal.billing * (1 - retain)
    return cal


# ---------------------------------------------------------------- F17
def learning_curve(hist=None):
    from cost_intel import ols
    hist = hist if hist is not None else pd.read_csv(MARKET / "building_history.csv")
    out = {}
    for col in ["cost_per_mw", "build_months"]:
        X = np.column_stack([np.ones(len(hist)), np.log(hist.sequence)])
        f = ols(np.log(hist[col]), X, ["log_a", "b"])
        b, se = f["beta"][1], f["coef"].std_error.iloc[1]
        tcrit = stats.t.ppf(0.975, f["n"] - 2)
        out[col] = {"a": float(np.exp(f["beta"][0])), "b": float(b), "rate": float(2 ** b), "rate_lo": float(2 ** (b - tcrit * se)),
                    "rate_hi": float(2 ** (b + tcrit * se)), "p": float(f["coef"].p_value.iloc[1]), "r2": f["r2"], "fit": f}
    return out, hist


def predict_curve(lc, n, col="cost_per_mw"):
    c = lc[col]
    return c["a"] * np.asarray(n, float) ** c["b"]


def program_learning(tables, controls):
    """This program's buildings in start order: committed value per unit of scale (a proxy for cost per MW)."""
    prj = tables.get("projects")
    if prj is None or "scale" not in prj.columns:
        return pd.DataFrame()
    pol = tables["po_lines"]
    po = tables["purchase_orders"]
    val = pol[pol.line_source == "CONTRACT"].merge(po[["po_id", "project_id"]], on="po_id").groupby("project_id").scheduled_value.sum()
    d = prj[["project_id", "short_name", "scale", "start"]].copy()
    d["contract_value"] = d.project_id.map(val)
    d = d.sort_values(["start", "project_id"]).reset_index(drop=True)
    d["sequence"] = np.arange(1, len(d) + 1)
    d["value_per_scale"] = d.contract_value / d.scale
    from cost_intel import ols
    f = ols(np.log(d.value_per_scale), np.column_stack([np.ones(len(d)), np.log(d.sequence)]), ["a", "b"])
    b = f["beta"][1]
    return d, {"rate": float(2 ** b), "p": float(f["coef"].p_value.iloc[1]), "r2": f["r2"], "n": len(d)}


# ---------------------------------------------------------------- F18
STAGE_ORDER = ["Auto-match (rules engine)", "Field verification", "PM approval", "Finance / AP", "Dispute resolution"]


def cycle_time(log, gates=None):
    log = log.copy()
    log["start"] = pd.to_datetime(log.start)
    log["end"] = pd.to_datetime(log.end, errors="coerce")
    data_date = max(log.start.max(), log.end.max())
    log["days"] = (log.end - log.start).dt.total_seconds() / 86400
    done = log.dropna(subset=["end"])
    stage = done.groupby("stage").days.agg(["mean", "median", lambda s: s.quantile(0.9), "count"])
    stage.columns = ["mean_days", "median_days", "p90_days", "bills"]
    stage = stage.reindex([s for s in STAGE_ORDER if s in stage.index])
    wip = log[log.end.isna()].groupby("stage").size().reindex(stage.index).fillna(0).astype(int)
    stage["in_progress_now"] = wip
    # bill-level cycle time (completed bills = all stages finished)
    per_bill = log.groupby("invoice_id").agg(start=("start", "min"), end=("end", "max"), open_=("end", lambda s: s.isna().any()))
    closed = per_bill[~per_bill.open_]
    closed_ct = (closed.end - closed.start).dt.total_seconds() / 86400
    window_days = (closed.end.max() - closed.end.min()).days or 1
    throughput = len(closed) / window_days * 30  # bills per month
    wip_now = int(per_bill.open_.sum())
    avg_wip = _avg_wip(per_bill, data_date)
    little_ct = avg_wip / (throughput / 30) if throughput else np.nan
    stage["total_days"] = stage.mean_days * stage.bills  # load: where the waiting time piles up
    stage["share_of_time_pct"] = 100 * stage.total_days / stage.total_days.sum()
    bottleneck = stage.total_days.idxmax()
    if gates is not None:
        g = closed_ct.rename("cycle_days").reset_index().merge(gates[["invoice_id", "gate"]], on="invoice_id", how="left")
        by_gate = g.groupby("gate").cycle_days.agg(["mean", "count"]).reset_index()
    else:
        by_gate = pd.DataFrame()
    return {"stages": stage.reset_index(), "cycle_days_mean": float(closed_ct.mean()), "cycle_days_p90": float(closed_ct.quantile(0.9)),
            "throughput_per_month": throughput, "wip_now": wip_now, "avg_wip": avg_wip, "little_cycle_days": little_ct,
            "bottleneck": bottleneck, "by_gate": by_gate, "per_bill": closed_ct, "log": log}


def _avg_wip(per_bill, data_date):
    s, e = per_bill.start.min(), data_date
    days = pd.date_range(s, e, freq="D")
    ends = per_bill.end.fillna(data_date)
    cnt = [((per_bill.start <= d) & (ends > d)).sum() for d in days]
    return float(np.mean(cnt))


# ---------------------------------------------------------------- F26
KEYWORD_FACTORS = [  # (regex on description, unit, kg per unit, kgCO2e per kg, material)
    (r"concrete|lining|coping", "cum", 2400.0, 0.11, "Concrete (low-strength, M10-M20)"),
    (r"LDPE|polyethylene film", "sqm", 0.184, 2.0, "LDPE film (200 micron)"),
    (r"brick", "sqm", 150.0, 0.24, "Brick pitching"),
    (r"steel|GI ", "rm", 3.6, 2.76, "Galvanized steel pipe"),
]


def carbon(tables, factors=None, spend_factor=0.30, equipment_codes=("E-105", "E-106", "E-107", "M-203", "M-204", "M-205")):
    factors = factors if factors is not None else pd.read_csv(MARKET / "carbon_factors.csv")
    pol = tables["po_lines"]
    po = tables["purchase_orders"]
    d = pol.merge(po[["po_id", "project_id"]], on="po_id")
    d["base_code"] = d.item_code
    co = tables["change_orders"]
    if "base_item_code" in co.columns:
        m = co.set_index("co_id").base_item_code
        d.loc[d.line_source == "CHANGE_ORDER", "base_code"] = d.loc[d.line_source == "CHANGE_ORDER", "item_code"].map(m)
    f = factors.set_index("item_code")
    d["kgco2e_per_unit"] = d.base_code.map(f.kgco2e_per_unit)
    d["material"] = d.base_code.map(f.material)
    import re
    for rx, unit, kg, fac, mat in KEYWORD_FACTORS:
        m = d.kgco2e_per_unit.isna() & d.unit.astype(str).str.lower().eq(unit) & d.description.astype(str).str.contains(rx, flags=re.I, regex=True)
        d.loc[m, "kgco2e_per_unit"] = kg * fac
        d.loc[m, "material"] = mat
    eq = d.base_code.isin(equipment_codes)
    d.loc[eq, "material"] = "Equipment (spend-based)"
    d["tco2e"] = np.where(eq, d.scheduled_value.fillna(d.contract_qty * d.contract_unit_rate) * spend_factor / 1000,
                          d.contract_qty * d.kgco2e_per_unit / 1000)
    d["method"] = np.where(eq, "spend-based", np.where(d.kgco2e_per_unit.notna(), "quantity x factor", "not covered"))
    covered = d[d.method != "not covered"]
    by_project = covered.groupby("project_id").agg(tco2e=("tco2e", "sum"), value=("scheduled_value", "sum")).reset_index()
    total_val = d.groupby("project_id").scheduled_value.sum()
    by_project["contract_value"] = by_project.project_id.map(total_val)
    by_project["t_per_musd"] = by_project.tco2e / (by_project.contract_value / 1e6)
    by_material = covered.groupby("material").tco2e.sum().sort_values(ascending=False).reset_index()
    coverage = covered.scheduled_value.sum() / d.scheduled_value.sum() if d.scheduled_value.sum() else 0
    return {"lines": d, "by_project": by_project, "by_material": by_material, "coverage": float(coverage),
            "total": float(covered.tco2e.sum())}
