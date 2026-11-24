"""
Project health score (0-100) built from four transparent sub-scores.

  Cost      budget headroom: +3% or more under budget = 100, falling to 0 at 5% over budget
  Schedule  the worse of: schedule index (1.00 = 100, 0.70 = 0) and weeks late (each week -6 points)
  Billing   60% first-pass rate + 40% billing accuracy (each 1% of billing at risk = -10 points)
  Cash      starts at 100; minus advance-recovery shortfall, flagged money already paid,
            overdue bills and slow payment (see CASH_RULES)
  Overall   30% cost + 30% schedule + 25% billing + 15% cash (re-weighted if a part has no data)
  Status    80+ Healthy, 60-79 Watch, below 60 At risk
  Red flags conditions that force attention WHATEVER the score (a good score elsewhere cannot hide them):
            finish forecast more than 4 weeks past the contract date, or forecast more than 3% over budget.
            Any red flag sets the status to At risk.
"""
import numpy as np
import pandas as pd

RED_FLAG_WEEKS_LATE = 4
RED_FLAG_OVER_BUDGET = 0.03
WEIGHTS = {"cost": 0.30, "schedule": 0.30, "billing": 0.25, "cash": 0.15}
CASH_RULES = [
    ("advance shortfall, % of advance", 2.0),
    ("flagged money already paid, % of billed", 20.0),
    ("overdue bills, % of billed", 5.0),
    ("average days beyond payment terms", 1.5),
]


def _clip(v):
    return float(np.clip(v, 0, 100)) if pd.notna(v) else np.nan


def status_of(score):
    if pd.isna(score):
        return "No data"
    return "Healthy" if score >= 80 else "Watch" if score >= 60 else "At risk"


def health_table(pt, cash):
    c = cash.set_index("project_id") if len(cash) else pd.DataFrame()
    rows = []
    for r in pt.itertuples():
        g = lambda col, default=np.nan: (c.at[r.project_id, col] if (len(c) and col in c.columns and r.project_id in c.index) else default)  # noqa: E731
        # cost
        cost = np.nan
        if pd.notna(getattr(r, "budget", np.nan)) and r.budget:
            vac_pct = 100 * r.vac / r.budget
            cost = _clip(100 * (vac_pct + 5) / 8)
        # schedule
        sched = np.nan
        parts = []
        if pd.notna(r.spi):
            parts.append(_clip(100 * (r.spi - 0.7) / 0.3))
        if pd.notna(getattr(r, "delay_weeks", np.nan)):
            parts.append(_clip(100 - 6 * r.delay_weeks))
        if parts:
            sched = min(parts)
        # billing
        billed = r.billed if pd.notna(r.billed) and r.billed else np.nan
        risk_pct = 100 * ((r.blocked or 0) + (r.to_check or 0)) / billed if pd.notna(billed) else np.nan
        bill = np.nan
        if pd.notna(getattr(r, "first_pass_pct", np.nan)):
            bill = _clip(0.6 * r.first_pass_pct + 0.4 * _clip(100 - 10 * (risk_pct if pd.notna(risk_pct) else 0)))
        # cash
        cashs = np.nan
        flags_cash = []
        if len(c) and r.project_id in c.index:
            pen = 0.0
            adv, short = g("advance_paid", 0), g("advance_shortfall", 0)
            if pd.notna(adv) and adv and pd.notna(short):
                pen += CASH_RULES[0][1] * 100 * short / adv
                if short > 1000:
                    flags_cash.append(("warn", f"Advance recovery ${short:,.0f} behind"))
            gross = g("billed_gross", np.nan)
            rec = g("paid_needs_recovery", 0)
            if pd.notna(rec) and pd.notna(gross) and gross:
                pen += CASH_RULES[1][1] * 100 * rec / gross
                if rec > 0:
                    flags_cash.append(("crit", f"${rec:,.0f} paid on flagged bills: recover"))
            od = g("overdue_amount", 0)
            if pd.notna(od) and pd.notna(gross) and gross:
                pen += CASH_RULES[2][1] * 100 * od / gross
                if od > 0:
                    flags_cash.append(("warn", f"${od:,.0f} unpaid past terms"))
            days, terms = g("avg_days_to_pay"), g("terms")
            if pd.notna(days) and pd.notna(terms):
                pen += CASH_RULES[3][1] * max(0, days - terms)
                if days > terms + 2:
                    flags_cash.append(("info", f"Paid in {days:.0f} days vs {terms:.0f}-day terms"))
            cashs = _clip(100 - pen)
        sub = {"cost": cost, "schedule": sched, "billing": bill, "cash": cashs}
        w = {k: WEIGHTS[k] for k, v in sub.items() if pd.notna(v)}
        overall = sum(sub[k] * w[k] for k in w) / sum(w.values()) if w else np.nan
        flags = []
        if pd.notna(cost) and r.vac < 0:
            flags.append(("crit", f"Forecast ${-r.vac:,.0f} over budget"))
        if pd.notna(getattr(r, "delay_weeks", np.nan)) and r.delay_weeks > 0:
            flags.append(("crit" if r.delay_weeks > 4 else "warn",
                          f"{r.delay_weeks:.0f} week{'s' if round(r.delay_weeks) != 1 else ''} late · penalty exposure ${r.ld_exposure:,.0f}"))
        if pd.notna(r.spi) and r.spi < 0.9:
            flags.append(("warn", f"Behind plan: {r.verified_pct:.0f}% done vs {r.planned_pct:.0f}% planned"))
        if getattr(r, "bills_on_hold", 0) and r.bills_on_hold > 0:
            flags.append(("crit", f"{int(r.bills_on_hold)} bill(s) flagged to hold · ${r.blocked:,.0f}"))
        if r.pending_co > 0:
            flags.append(("info", f"${r.pending_co:,.0f} change orders pending"))
        flags += flags_cash
        order = {"crit": 0, "warn": 1, "info": 2}
        flags.sort(key=lambda f: order[f[0]])
        red = []
        if pd.notna(getattr(r, "delay_weeks", np.nan)) and r.delay_weeks > RED_FLAG_WEEKS_LATE:
            red.append(f"Finish {r.delay_weeks:.0f} weeks past the contract date")
        if pd.notna(cost) and pd.notna(getattr(r, "budget", np.nan)) and r.budget and -r.vac / r.budget > RED_FLAG_OVER_BUDGET:
            red.append(f"Forecast {100 * -r.vac / r.budget:.1f}% over budget")
        status = "At risk" if red else status_of(overall)
        rows.append({"project_id": r.project_id, **{f"{k}_score": v for k, v in sub.items()},
                     "health": overall, "health_status": status, "risk_pct": risk_pct, "alerts": flags, "red_flags": red})
    return pd.DataFrame(rows)
