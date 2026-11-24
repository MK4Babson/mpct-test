"""
Generate a synthetic multi-prime data center construction commercial dataset.

What it produces (data/raw/*.csv):
  projects, primes             - fictional buildings and prime contractors
  purchase_orders, po_lines    - contract PO per (project, prime) + approved change-order lines
  change_orders                - CO log with APPROVED / PENDING / REJECTED status
  tm_preapprovals              - pre-approved time-and-materials hours per period
  goods_receipts               - field-verified work in place (the "receipt" leg of the 3-way match)
  invoices, invoice_lines      - monthly pay applications from each prime
  ground_truth_errors          - every error deliberately planted (used ONLY by evaluate.py)

Realism choices:
  * Field verification carries small measurement noise (+/-1%), so a perfectly honest
    invoice can still show billed > verified. This makes tolerance-setting a real trade-off.
  * Planted errors vary in size (some are tiny and hide inside the noise on purpose).

ALL DATA IS SYNTHETIC.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd

from config import (SEED, PERIODS, PROJECTS, PRIMES, RATE_CARDS, TM_RATES, BASE_HEADCOUNT,
                    WORKDAYS_PER_MONTH, RETAINAGE_RATE, PER_DIEM_CAP)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
rng = np.random.default_rng(SEED)
rrng = np.random.default_rng(SEED + 7919)  # separate stream for risk drivers

# Error-planting probabilities
P = {
    "RATE_OVER_CONTRACT": 0.02,      # per unit-price SOV line
    "QTY_OVER_RECEIPT": 0.03,        # per unit-price SOV line
    "PCT_COMPLETE_OVERBILL": 0.12,   # per LS line being billed
    "EXTENSION_ERROR": 0.008,        # per work line (qty x rate != amount)
    "UNAPPROVED_CO": 0.06,           # per invoice
    "TM_OVER_APPROVED": 0.06,        # per T&M line
    "PER_DIEM_OVER_POLICY": 0.07,    # per per-diem line
    "RETAINAGE_ERROR": 0.04,         # per invoice
    "DUPLICATE_INVOICE": 0.009,      # per invoice: rare (about one per program), as in practice
}


DUP_TARGET = ("WTX-A2", "PRM-E", "2026-05")

# ---- Risk drivers (assumptions about how billing errors behave; used by the ML risk model) ----
# Error probabilities are scaled per bill by observable-in-principle conditions, on a separate random stream:
#   contractor billing habit (most crews careful, a few habitual), first two bills of a contract (set-up errors),
#   steep progress months (rushed bills), close-out pressure (>75% billed), a pending change order on the PO,
#   and an error on the previous bill (habits persist). Set MPCT_RISK_DRIVERS=0 to plant errors uniformly.
RISK_DRIVERS = os.environ.get("MPCT_RISK_DRIVERS", "1") == "1"
HABIT_LEVELS, HABIT_P = [0.5, 0.8, 1.0, 1.4, 2.4], [0.20, 0.30, 0.25, 0.15, 0.10]
RISK_NORMALIZER = 2.2  # keeps the overall error rate close to the base probabilities  # project, prime, period of the planted resubmission


def qty_decimals(unit):
    return 1 if unit == "TON" else 0


def s_curve(month_idx, start, duration):
    """Cumulative fraction complete at END of month_idx (logistic S-curve)."""
    if month_idx < start:
        return 0.0
    t = (month_idx - start + 1) / duration
    if t >= 1:
        return 1.0
    x = 1 / (1 + np.exp(-10 * (t - 0.5)))
    x0 = 1 / (1 + np.exp(5))
    x1 = 1 / (1 + np.exp(-5))
    return float((x - x0) / (x1 - x0))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate the synthetic dataset")
    ap.add_argument("--out", default=str(RAW), help="output folder")
    ap.add_argument("--clean", action="store_true",
                    help="plant no errors (a 'clean' dataset to practise the stress test on)")
    args = ap.parse_args()
    if args.clean:
        for k in P:
            P[k] = 0.0
    raw_dir = Path(args.out)
    raw_dir.mkdir(parents=True, exist_ok=True)
    projects = pd.DataFrame(PROJECTS)
    primes = pd.DataFrame(PRIMES)

    po_rows, po_line_rows, co_rows, tm_rows, grn_rows = [], [], [], [], []
    inv_rows, line_rows, gt_rows = [], [], []
    inv_seq = 1000

    for proj in PROJECTS:
        for prime in PRIMES:
            pid, prm = proj["project_id"], prime["prime_id"]
            po_id = f"PO-{pid}-{prm[-1]}"
            po_rows.append({"po_id": po_id, "project_id": pid, "prime_id": prm,
                            "contract_type": "Unit price + lump sum", "retainage_rate": RETAINAGE_RATE})

            # ---- Contract SOV lines ----
            # Each prime prices each building a little differently (that's what benchmarking finds)
            ps = proj.get("start", 0)
            price_factor = float(rng.uniform(0.96, 1.10))
            sov = []
            for code, desc, unit, base_qty, rate, start, dur in RATE_CARDS[prm]:
                if unit == "LS":
                    qty = 1
                    sv = round(rate * proj["scale"] * price_factor, 2)
                    unit_rate = sv
                else:
                    qty = round(base_qty * proj["scale"], qty_decimals(unit))
                    unit_rate = round(rate * price_factor * float(rng.uniform(0.98, 1.03)), 2)
                # stretch the schedule so buildings are mid-construction at the end of the demo window
                sov.append(dict(item_code=code, description=desc, unit=unit, qty=qty,
                                unit_rate=unit_rate, start=round(start * 1.5), dur=round(dur * 1.7)))
                po_line_rows.append({"po_id": po_id, "item_code": code, "line_source": "CONTRACT",
                                     "co_id": None, "description": desc, "unit": unit,
                                     "contract_qty": qty, "contract_unit_rate": unit_rate,
                                     "scheduled_value": round(qty * unit_rate, 2)})

            # ---- Change orders (some approved, some not) ----
            cos = []
            n_cos = rng.integers(2, 5)
            for k in range(n_cos):
                base = sov[rng.integers(0, len([s for s in sov if s["unit"] != "LS"]))]
                status = rng.choice(["APPROVED", "PENDING", "REJECTED"], p=[0.6, 0.25, 0.15])
                eff = min(len(PERIODS) - 1, ps + int(rng.integers(1, 5)))  # effective month index
                co_id = f"CO-{pid}-{prm[-1]}-{k+1:02d}"
                co_qty = round(base["qty"] * rng.uniform(0.04, 0.12), qty_decimals(base["unit"]))
                if base["unit"] == "EA":
                    co_qty = max(1, int(co_qty))
                co_rate = round(base["unit_rate"] * rng.uniform(1.0, 1.08), 2)
                co = dict(co_id=co_id, po_id=po_id, project_id=pid, prime_id=prm,
                          base_item_code=base["item_code"],
                          description=f"Added scope: {base['description']}",
                          unit=base["unit"], co_qty=co_qty, co_unit_rate=co_rate,
                          co_amount=round(co_qty * co_rate, 2), status=status,
                          approval_period=PERIODS[eff] if status == "APPROVED" else None,
                          eff_idx=eff)
                cos.append(co)
                co_rows.append({k2: v for k2, v in co.items() if k2 != "eff_idx"})
                if status == "APPROVED":
                    po_line_rows.append({"po_id": po_id, "item_code": co_id, "line_source": "CHANGE_ORDER",
                                         "co_id": co_id, "description": co["description"], "unit": co["unit"],
                                         "contract_qty": co_qty, "contract_unit_rate": co_rate,
                                         "scheduled_value": co["co_amount"]})

            # ---- Monthly billing ----
            habit = float(rrng.choice(HABIT_LEVELS, p=HABIT_P))
            prev_err = False
            tot_value = sum(s_["qty"] * s_["unit_rate"] for s_ in sov) or 1.0
            for m, period in enumerate(PERIODS):
                if m < ps:
                    continue  # this building hasn't started yet
                lm = m - ps   # month number within this project
                inv_seq += 1
                invoice_id = f"INV-{inv_seq}"
                invoice_number = f"{prm[-1]}{pid[-2:]}-{m+1:03d}"
                inv_date = pd.Timestamp(period + "-25") + pd.Timedelta(days=int(rng.integers(0, 6)))
                lines = []
                gt_inv = []  # (line_no or None, error_type, overbilled_amount)
                if RISK_DRIVERS:
                    inc = sum((s_curve(lm, s_["start"], s_["dur"]) - (s_curve(lm - 1, s_["start"], s_["dur"]) if lm > 0 else 0.0))
                              * s_["qty"] * s_["unit_rate"] for s_ in sov) / tot_value
                    cum = sum(s_curve(lm, s_["start"], s_["dur"]) * s_["qty"] * s_["unit_rate"] for s_ in sov) / tot_value
                    pending = any(c_["status"] == "PENDING" and c_["eff_idx"] <= m for c_ in cos)
                    R = (habit * (1.5 if lm < 2 else 1.0) * (1.3 if inc > 0.06 else 1.0) * (1.3 if cum > 0.75 else 1.0)
                         * (1.5 if pending else 1.0) * (1.35 if prev_err else 1.0)) / RISK_NORMALIZER
                else:
                    R = 1.0
                pr = lambda k: min(0.85, P[k] * R)  # noqa: E731

                # SOV lines
                for s in sov:
                    c_now = s_curve(lm, s["start"], s["dur"])
                    c_prev = s_curve(lm - 1, s["start"], s["dur"]) if lm > 0 else 0.0
                    if c_now - c_prev <= 1e-9:
                        continue
                    grn_id = f"GRN-{pid}-{prm[-1]}-{s['item_code']}-{period}"
                    if s["unit"] == "LS":
                        true_cum = round(c_now * 100, 1)
                        prev_cum = round(c_prev * 100, 1)
                        ver_cum = float(np.clip(round((true_cum + rng.normal(0, 0.4)) * 2) / 2, 0, 100))
                        grn_rows.append({"grn_id": grn_id, "po_id": po_id, "item_code": s["item_code"],
                                         "period": period, "verified_qty": None,
                                         "verified_pct_complete_cum": ver_cum})
                        billed_cum = true_cum
                        err = None
                        if rng.random() < pr("PCT_COMPLETE_OVERBILL") and true_cum < 95:
                            billed_cum = round(min(100.0, true_cum + rng.uniform(1.5, 20)), 1)
                            err = "PCT_COMPLETE_OVERBILL"
                        amount = round(s["unit_rate"] * (billed_cum - prev_cum) / 100, 2)
                        lines.append(dict(line_type="SOV", ref_id=s["item_code"], description=s["description"],
                                          unit="LS", qty=None, unit_rate=None, amount=amount,
                                          pct_complete_prev=prev_cum, pct_complete_cum=billed_cum,
                                          grn_id=grn_id, _err=err,
                                          _over=round(s["unit_rate"] * (billed_cum - true_cum) / 100, 2)))
                    else:
                        d = qty_decimals(s["unit"])
                        true_period = round(s["qty"] * c_now, d) - round(s["qty"] * c_prev, d)
                        true_period = round(true_period, d)
                        if true_period <= 0:
                            continue
                        if s["unit"] == "EA":
                            ver = true_period  # discrete items are counted exactly
                        else:
                            ver = round(true_period * (1 + float(np.clip(rng.normal(0, 0.004), -0.01, 0.01))), d)
                        grn_rows.append({"grn_id": grn_id, "po_id": po_id, "item_code": s["item_code"],
                                         "period": period, "verified_qty": ver,
                                         "verified_pct_complete_cum": None})
                        qty, rate, err, over = true_period, s["unit_rate"], None, 0.0
                        r = rng.random()
                        if r < pr("RATE_OVER_CONTRACT"):
                            rate = round(s["unit_rate"] * rng.uniform(1.02, 1.15), 2)
                            err, over = "RATE_OVER_CONTRACT", round((rate - s["unit_rate"]) * qty, 2)
                        elif r < pr("RATE_OVER_CONTRACT") + pr("QTY_OVER_RECEIPT"):
                            f = rng.uniform(1.015, 1.20)
                            qty = round(true_period * f, d)
                            if s["unit"] == "EA" and qty <= true_period:
                                qty = true_period + 1
                            err, over = "QTY_OVER_RECEIPT", round((qty - ver) * rate, 2)
                        lines.append(dict(line_type="SOV", ref_id=s["item_code"], description=s["description"],
                                          unit=s["unit"], qty=qty, unit_rate=rate, amount=round(qty * rate, 2),
                                          pct_complete_prev=None, pct_complete_cum=None, grn_id=grn_id,
                                          _err=err, _over=over))

                # Approved change-order work (billed in the two months after approval)
                for co in cos:
                    if co["status"] == "APPROVED" and co["eff_idx"] <= m <= co["eff_idx"] + 1:
                        d = qty_decimals(co["unit"])
                        q = round(co["co_qty"] / 2, d) if m == co["eff_idx"] else round(co["co_qty"] - round(co["co_qty"] / 2, d), d)
                        if q <= 0:
                            continue
                        lines.append(dict(line_type="CO", ref_id=co["co_id"], description=co["description"],
                                          unit=co["unit"], qty=q, unit_rate=co["co_unit_rate"],
                                          amount=round(q * co["co_unit_rate"], 2), pct_complete_prev=None,
                                          pct_complete_cum=None, grn_id=None, _err=None, _over=0.0))

                # Planted: billing a change order that is not approved
                if rng.random() < pr("UNAPPROVED_CO"):
                    bad = [c for c in cos if c["status"] != "APPROVED"]
                    if bad:
                        c = bad[int(rng.integers(0, len(bad)))]
                        ref, unit, rate = c["co_id"], c["unit"], c["co_unit_rate"]
                        q = round(c["co_qty"] * rng.uniform(0.3, 1.0), qty_decimals(unit))
                        if unit == "EA":
                            q = max(1, int(q))
                        desc = c["description"]
                    else:
                        ref, unit, q, rate = f"CO-{pid}-{prm[-1]}-99", "LS", 1, round(rng.uniform(40000, 250000), 2)
                        desc = "Field directive - extra work (no CO on file)"
                    lines.append(dict(line_type="CO", ref_id=ref, description=desc, unit=unit, qty=q,
                                      unit_rate=rate, amount=round(q * rate, 2), pct_complete_prev=None,
                                      pct_complete_cum=None, grn_id=None, _err="UNAPPROVED_CO",
                                      _over=round(q * rate, 2)))

                # T&M lines against pre-approvals
                for craft, tm_rate in TM_RATES[prm]:
                    appr_id = f"TMA-{pid}-{prm[-1]}-{period}-{craft[:3].upper()}"
                    approved = int(rng.integers(8, 40) * 10)
                    tm_rows.append({"approval_id": appr_id, "po_id": po_id, "period": period, "craft": craft,
                                    "approved_hours": approved, "approved_rate": tm_rate})
                    hours = int(approved * rng.uniform(0.6, 1.0))
                    err, over = None, 0.0
                    if rng.random() < pr("TM_OVER_APPROVED"):
                        hours = int(np.ceil(approved * rng.uniform(1.05, 1.40)))
                        err, over = "TM_OVER_APPROVED", round((hours - approved) * tm_rate, 2)
                    lines.append(dict(line_type="TM", ref_id=appr_id, description=f"T&M - {craft}", unit="HR",
                                      qty=hours, unit_rate=tm_rate, amount=round(hours * tm_rate, 2),
                                      pct_complete_prev=None, pct_complete_cum=None, grn_id=None,
                                      _err=err, _over=over))

                # Per diem line
                ramp = 0.5 + 0.5 * s_curve(lm, 0, 4)
                workers = int(BASE_HEADCOUNT[prm] * proj["scale"] * ramp * rng.uniform(0.9, 1.1))
                wd = workers * WORKDAYS_PER_MONTH
                pd_rate = float(rng.choice([PER_DIEM_CAP, 172.00, 165.00, 160.00]))
                err, over = None, 0.0
                if rng.random() < pr("PER_DIEM_OVER_POLICY"):
                    pd_rate = round(PER_DIEM_CAP * rng.uniform(1.05, 1.35), 2)
                    err, over = "PER_DIEM_OVER_POLICY", round((pd_rate - PER_DIEM_CAP) * wd, 2)
                lines.append(dict(line_type="PER_DIEM", ref_id="PER-DIEM", description=f"Per diem - {workers} workers",
                                  unit="WORKER-DAY", qty=wd, unit_rate=pd_rate, amount=round(wd * pd_rate, 2),
                                  pct_complete_prev=None, pct_complete_cum=None, grn_id=None,
                                  _err=err, _over=over))

                # Planted arithmetic error on one random work line (qty x rate != amount)
                for ln in lines:
                    if ln["_err"] is None and ln["qty"] is not None and rng.random() < pr("EXTENSION_ERROR"):
                        bump = round(float(rng.uniform(500, 25000)), 2)
                        ln["amount"] = round(ln["amount"] + bump, 2)
                        ln["_err"], ln["_over"] = "EXTENSION_ERROR", bump
                        break

                # Number the lines, record ground truth
                for i, ln in enumerate(lines, start=1):
                    ln["line_no"] = i
                    if ln["_err"]:
                        gt_inv.append((i, ln["_err"], ln["_over"]))

                gross = round(sum(l["amount"] for l in lines), 2)
                ret_base = round(sum(l["amount"] for l in lines if l["line_type"] in ("SOV", "CO")), 2)
                retainage = round(ret_base * RETAINAGE_RATE, 2)
                if rng.random() < pr("RETAINAGE_ERROR"):
                    wrong = round(ret_base * 0.05, 2) if rng.random() < 0.6 else round(gross * RETAINAGE_RATE * 0.5, 2)
                    gt_inv.append((None, "RETAINAGE_ERROR", round(retainage - wrong, 2)))
                    retainage = wrong
                header = {"invoice_id": invoice_id, "invoice_number": invoice_number, "po_id": po_id,
                          "project_id": pid, "prime_id": prm, "period": period,
                          "invoice_date": inv_date.date().isoformat(), "gross_amount": gross,
                          "retainage_withheld": retainage, "net_amount": round(gross - retainage, 2)}
                inv_rows.append(header)
                for ln in lines:
                    line_rows.append({"invoice_id": invoice_id, **{k: v for k, v in ln.items() if not k.startswith("_")}})
                prev_err = len(gt_inv) > 0
                for ln_no, et, over in gt_inv:
                    gt_rows.append({"invoice_id": invoice_id, "line_no": ln_no, "error_type": et,
                                    "overbilled_amount": over})

                # Planted: duplicate submission of the same pay application
                # one realistic resubmission is always planted: a copy with a "-R" number, sent ~12 days later
                forced = P["DUPLICATE_INVOICE"] > 0 and (pid, prm, period) == DUP_TARGET
                if rng.random() < P["DUPLICATE_INVOICE"] or forced:
                    inv_seq += 1
                    dup_id = f"INV-{inv_seq}"
                    dup = dict(header, invoice_id=dup_id,
                               invoice_number=(invoice_number + "-R") if forced else (invoice_number + rng.choice(["-R", "A", "-1", ""]) if rng.random() < 0.8 else f"{prm[-1]}{pid[-2:]}-{m+51:03d}"),
                               invoice_date=(inv_date + pd.Timedelta(days=int(rng.integers(3, 21)))).date().isoformat())
                    inv_rows.append(dup)
                    for ln in lines:
                        line_rows.append({"invoice_id": dup_id, **{k: v for k, v in ln.items() if not k.startswith("_")}})
                    gt_rows.append({"invoice_id": dup_id, "line_no": None, "error_type": "DUPLICATE_INVOICE",
                                    "overbilled_amount": header["net_amount"]})
                    for ln_no, et, over in gt_inv:  # the copied lines carry the same errors
                        gt_rows.append({"invoice_id": dup_id, "line_no": ln_no, "error_type": et,
                                        "overbilled_amount": over})

    out = {
        "projects": projects, "primes": primes,
        "purchase_orders": pd.DataFrame(po_rows), "po_lines": pd.DataFrame(po_line_rows),
        "change_orders": pd.DataFrame(co_rows), "tm_preapprovals": pd.DataFrame(tm_rows),
        "goods_receipts": pd.DataFrame(grn_rows), "invoices": pd.DataFrame(inv_rows),
        "invoice_lines": pd.DataFrame(line_rows),
        "ground_truth_errors": pd.DataFrame(gt_rows, columns=["invoice_id", "line_no", "error_type", "overbilled_amount"]),
    }
    for name, df in out.items():
        df.to_csv(raw_dir / f"{name}.csv", index=False)
        print(f"{name:22s} {len(df):6d} rows")

    gt = out["ground_truth_errors"]
    if gt.empty:
        print("\nNo errors planted (clean dataset).")
        return
    print("\nPlanted errors by type:")
    print(gt.groupby("error_type").agg(count=("error_type", "size"),
                                       overbilled=("overbilled_amount", "sum")).round(0).to_string())


if __name__ == "__main__":
    main()
