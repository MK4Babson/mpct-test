"""
Create the optional program-control tables (project_controls, baseline_plan, payments)
for the synthetic data center demo, so the manager views have budget, schedule and cash data.

Usage:  python src/build_controls.py --data data/raw
The values are crafted: each building gets a schedule-performance factor and a budget,
and payments follow typical net-30 practice with a few late and a few short-paid bills.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_engine import run
from schema import read_folder, coerce
from manager import progress_by_period

ROOT = Path(__file__).resolve().parents[1]
# >1 means the plan expected more progress than was achieved (the building is behind)
SCHEDULE_FACTOR = {"WTX-A1": 1.00, "WTX-A2": 1.05, "WTX-B1": 1.14, "WTX-B2": 1.00, "WTX-C1": 1.22, "WTX-C2": 0.98}
CONTINGENCY = {"WTX-A1": 0.06, "WTX-A2": 0.05, "WTX-B1": -0.01, "WTX-B2": 0.04, "WTX-C1": 0.03, "WTX-C2": 0.07}
SHORT = {"WTX-A1": "Mesa A1", "WTX-A2": "Mesa A2", "WTX-B1": "Ridge B1", "WTX-B2": "Ridge B2",
         "WTX-C1": "Plains C1", "WTX-C2": "Plains C2"}
DATA_DATE = "2026-08-31"
# fictional campuses: illustrative West Texas coordinates
COORDS = {"WTX-A1": (32.3810, -100.8620), "WTX-A2": (32.3858, -100.8551), "WTX-B1": (33.1210, -101.5790),
          "WTX-B2": (33.1262, -101.5718), "WTX-C1": (34.4205, -101.8010), "WTX-C2": (34.4251, -101.7932)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    data = Path(a.data)
    rng = np.random.default_rng(a.seed)
    raw = read_folder(data)
    tables = {k: coerce(v, k) for k, v in raw.items()}
    _, gates = run(tables=tables)
    prog = progress_by_period(tables, gates)
    periods = sorted(prog.period.unique())
    co = tables["change_orders"].merge(tables["purchase_orders"][["po_id", "project_id"]], on="po_id", suffixes=("", "_po"))
    pending = co[co.status == "PENDING"].groupby("project_id").co_amount.sum()

    pc, bp = [], []
    for p, g in prog.groupby("project_id"):
        g = g.sort_values("period")
        f = SCHEDULE_FACTOR.get(p, 1.0)
        planned = (g.verified_pct.fillna(0) * f).clip(upper=100).round(1).tolist()
        pers = g.period.tolist()
        started = [x for x, v in zip(pers, planned) if v > 0]
        # extend the plan past the data date at the recent planned pace until it reaches 100%
        pace = max(2.0, (planned[-1] - planned[-4]) / 3) if len(planned) >= 4 else 8.0
        cur, per = planned[-1], pd.Period(pers[-1], "M")
        while cur < 100:
            per += 1
            cur = min(100.0, round(cur + pace, 1))
            pers.append(str(per))
            planned.append(cur)
        for x, v in zip(pers, planned):
            bp.append({"project_id": p, "period": x, "planned_pct_cum": v})
        finish = pd.Period(pers[planned.index(100.0)], "M").end_time.date()
        committed = g.committed.iloc[0]
        c = CONTINGENCY.get(p, 0.05)
        budget = (committed + pending.get(p, 0)) * (1 + c) if c < 0 else committed * (1 + c)
        pc.append({"project_id": p, "short_name": SHORT.get(p, p), "approved_budget": round(budget, -3),
                   "planned_start": f"{started[0] if started else pers[0]}-01", "planned_finish": finish,
                   "forecast_finish": None, "data_date": DATA_DATE,
                   "ld_rate_pct_per_week": 0.25, "ld_cap_pct": 5.0, "advance_paid": 0,
                   "advance_recovery_start_pct": None, "advance_recovery_end_pct": None, "payment_terms_days": 30})

    # payments: net 30; bills with errors were sometimes short-paid by the AP team (caught by hand)
    inv = tables["invoices"].copy()
    truth = raw.get("ground_truth_errors", pd.DataFrame(columns=["invoice_id", "error_type", "overbilled_amount"]))
    over = truth.assign(o=pd.to_numeric(truth.overbilled_amount, errors="coerce")).groupby("invoice_id").o.sum()
    dups = set(truth[truth.error_type == "DUPLICATE_INVOICE"].invoice_id)
    rows = []
    last = periods[-1] if periods else None
    for r in inv.itertuples():
        d = pd.Timestamp(r.invoice_date)
        if r.invoice_id in dups or r.period == last:
            rows.append({"invoice_id": r.invoice_id, "submitted_date": d.date(), "paid_date": None, "amount_paid": None,
                         "advance_recovered": 0, "amount_withheld": 0,
                         "status": "Rejected as duplicate" if r.invoice_id in dups else "In review"})
            continue
        # one contractor's July bills at Ridge are stuck in approval (overdue)
        if r.period == "2026-07" and r.project_id in ("WTX-B1", "WTX-B2") and r.prime_id == "PRM-M":
            rows.append({"invoice_id": r.invoice_id, "submitted_date": d.date(), "paid_date": None, "amount_paid": None,
                         "advance_recovered": 0, "amount_withheld": 0, "status": "Stuck in approval"})
            continue
        withheld = float(over.get(r.invoice_id, 0)) if rng.random() < 0.55 else 0.0
        days = int(rng.integers(14, 27)) + (7 if withheld else 0) + (9 if r.project_id.startswith("WTX-C") else 0)
        rows.append({"invoice_id": r.invoice_id, "submitted_date": d.date(), "paid_date": (d + pd.Timedelta(days=days)).date(),
                     "amount_paid": round(r.net_amount - withheld, 2), "advance_recovered": 0,
                     "amount_withheld": round(withheld, 2), "status": "Paid"})

    pd.DataFrame(pc).to_csv(data / "project_controls.csv", index=False)
    pd.DataFrame(bp).to_csv(data / "baseline_plan.csv", index=False)
    pd.DataFrame(rows).to_csv(data / "payments.csv", index=False)
    pr = raw["projects"].copy()
    pr["short_name"] = pr.project_id.map(SHORT)
    pr["latitude"] = pr.project_id.map(lambda x: COORDS.get(x, (None, None))[0])
    pr["longitude"] = pr.project_id.map(lambda x: COORDS.get(x, (None, None))[1])
    pr["location"] = pr.campus.str.replace(" (fictional)", "", regex=False) + ", West Texas, USA (illustrative location)"
    pr["structure"] = "datacenter"
    pr["project_name"] = "Hyperscale data center building " + pr.project_id.str[-2:]
    pr.to_csv(data / "projects.csv", index=False)
    print(f"wrote project_controls, baseline_plan, payments to {data}")


if __name__ == "__main__":
    main()
