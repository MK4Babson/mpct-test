"""
Stress-test the engine on ANY data folder, including real data with no answer key.

Real invoices don't come with a list of "the errors in here", so we create one:
  1. Run the engine on your data as-is            -> baseline (pre-existing exceptions)
  2. Copy the data and plant known errors in lines the baseline considered clean
  3. Run the engine again on the copy             -> did it catch what we planted?
  4. Anything new that we did NOT plant           -> a false alarm caused by the engine

Usage:
  python src/inject_errors.py --data data/my_project --n 40
  python src/inject_errors.py --data data/my_project --n 40 --subtle 0.3   # 30% tiny errors

Writes the modified copy (with ground_truth_errors.csv) to <data>_stress/ and a report to
<data>_stress/stress_test_report.md
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from run_engine import run
from schema import SCHEMA, coerce, read_folder
from config import PER_DIEM_CAP

TYPES = ["RATE_OVER_CONTRACT", "QTY_OVER_RECEIPT", "PCT_COMPLETE_OVERBILL", "EXTENSION_ERROR",
         "UNAPPROVED_CO", "TM_OVER_APPROVED", "PER_DIEM_OVER_POLICY", "RETAINAGE_ERROR",
         "DUPLICATE_INVOICE"]


def keyset(ex):
    return set(zip(ex["invoice_id"], ex["line_no"].fillna(-1).astype(int), ex["rule"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=40, help="how many errors to plant")
    ap.add_argument("--subtle", type=float, default=0.0,
                    help="share of planted errors that are deliberately tiny (0-1)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out or str(Path(args.data).resolve()) + "_stress")
    out.mkdir(parents=True, exist_ok=True)

    raw = read_folder(args.data)
    t = {k: coerce(v, k) for k, v in raw.items()}
    base_ex, base_gates = run(tables=t)
    base_keys = keyset(base_ex)
    flagged_lines = {(i, l) for i, l, _ in base_keys if l != -1}
    flagged_invoice_rules = {(i, r) for i, l, r in base_keys if l == -1}

    inv, lines = t["invoices"].copy(), t["invoice_lines"].copy()
    lines = lines.merge(inv[["invoice_id", "po_id"]], on="invoice_id", how="left")
    po = t["purchase_orders"].set_index("po_id")
    pol = t["po_lines"]
    sv = {(a, b): c for a, b, c in zip(pol["po_id"], pol["item_code"], pol["scheduled_value"])}
    rate_map = {(a, b): c for a, b, c in zip(pol["po_id"], pol["item_code"], pol["contract_unit_rate"])}
    grn = t["goods_receipts"].set_index("grn_id")
    tm = t["tm_preapprovals"].set_index("approval_id")
    used_lines, used_inv = set(), set()
    truth, new_lines, dup_invoices, dup_lines = [], [], [], []

    def clean(row):
        k = (row.invoice_id, int(row.line_no))
        return k not in flagged_lines and k not in used_lines

    def candidates(mask):
        c = lines[mask]
        return [r for r in c.itertuples() if clean(r)]

    def adjust_header(invoice_id, delta, retainable):
        i = inv.index[inv.invoice_id == invoice_id][0]
        rr = po.loc[inv.at[i, "po_id"], "retainage_rate"]
        inv.at[i, "gross_amount"] = round(inv.at[i, "gross_amount"] + delta, 2)
        if retainable:
            inv.at[i, "retainage_withheld"] = round(inv.at[i, "retainage_withheld"] + delta * rr, 2)
        inv.at[i, "net_amount"] = round(inv.at[i, "gross_amount"] - inv.at[i, "retainage_withheld"], 2)

    def size(lo_hi_normal, lo_hi_subtle):
        tiny = rng.random() < args.subtle
        lo, hi = lo_hi_subtle if tiny else lo_hi_normal
        return float(rng.uniform(lo, hi)), tiny

    is_ls = lines["unit"].astype(str).str.upper() == "LS"
    pools = {
        "RATE_OVER_CONTRACT": lambda: candidates((lines.line_type == "SOV") & ~is_ls & lines.qty.notna()
                                                 & np.array([(p, r) in rate_map for p, r in zip(lines.po_id, lines.ref_id)], dtype=bool)),
        "QTY_OVER_RECEIPT": lambda: candidates((lines.line_type == "SOV") & ~is_ls & lines.qty.notna()
                                               & lines.grn_id.isin(grn.index)),
        "PCT_COMPLETE_OVERBILL": lambda: candidates((lines.line_type == "SOV") & is_ls & lines.grn_id.isin(grn.index)),
        "EXTENSION_ERROR": lambda: candidates(lines.qty.notna() & lines.unit_rate.notna()),
        "TM_OVER_APPROVED": lambda: candidates((lines.line_type == "TM") & lines.ref_id.isin(tm.index)),
        "PER_DIEM_OVER_POLICY": lambda: candidates(lines.line_type == "PER_DIEM"),
    }

    plan = list(rng.choice(TYPES, size=args.n))
    skipped = []
    for et in plan:
        if et in pools:
            pool = pools[et]()
            if not pool:
                skipped.append(et)
                continue
            r = pool[int(rng.integers(0, len(pool)))]
            idx = lines.index[(lines.invoice_id == r.invoice_id) & (lines.line_no == r.line_no)][0]
            old_amt = lines.at[idx, "amount"]
            if et == "RATE_OVER_CONTRACT":
                f, tiny = size((1.03, 1.15), (1.001, 1.005))
                lines.at[idx, "unit_rate"] = round(rate_map[(r.po_id, r.ref_id)] * f, 2)
                lines.at[idx, "amount"] = round(lines.at[idx, "qty"] * lines.at[idx, "unit_rate"], 2)
                over = (lines.at[idx, "unit_rate"] - rate_map[(r.po_id, r.ref_id)]) * lines.at[idx, "qty"]
            elif et == "QTY_OVER_RECEIPT":
                ver = grn.at[r.grn_id, "verified_qty"]
                if pd.isna(ver) or ver <= 0:
                    skipped.append(et)
                    continue
                f, tiny = size((1.03, 1.20), (1.002, 1.012))
                lines.at[idx, "qty"] = round(ver * f, 2)
                lines.at[idx, "amount"] = round(lines.at[idx, "qty"] * lines.at[idx, "unit_rate"], 2)
                over = (lines.at[idx, "qty"] - ver) * lines.at[idx, "unit_rate"]
            elif et == "PCT_COMPLETE_OVERBILL":
                ver = grn.at[r.grn_id, "verified_pct_complete_cum"]
                if pd.isna(ver) or ver >= 97 or (r.po_id, r.ref_id) not in sv:
                    skipped.append(et)
                    continue
                f, tiny = size((3, 15), (0.2, 0.9))
                lines.at[idx, "pct_complete_cum"] = round(min(100.0, ver + f), 1)
                lines.at[idx, "amount"] = round(sv[(r.po_id, r.ref_id)] *
                                                (lines.at[idx, "pct_complete_cum"] - lines.at[idx, "pct_complete_prev"]) / 100, 2)
                over = sv[(r.po_id, r.ref_id)] * (lines.at[idx, "pct_complete_cum"] - ver) / 100
            elif et == "EXTENSION_ERROR":
                f, tiny = size((0.01, 0.05), (0.0, 0.0))
                bump = max(100.0, abs(old_amt) * f) if not tiny else 0.40  # 40 cents = below $1 rounding tolerance
                lines.at[idx, "amount"] = round(old_amt + bump, 2)
                over = bump
            elif et == "TM_OVER_APPROVED":
                f, tiny = size((1.12, 1.40), (1.01, 1.05))
                appr = tm.at[r.ref_id, "approved_hours"]
                lines.at[idx, "qty"] = float(np.ceil(appr * f))
                over = (lines.at[idx, "qty"] - appr) * lines.at[idx, "unit_rate"]
                lines.at[idx, "amount"] = round(lines.at[idx, "qty"] * lines.at[idx, "unit_rate"], 2)
            elif et == "PER_DIEM_OVER_POLICY":
                f, tiny = size((1.05, 1.30), (1.001, 1.01))
                lines.at[idx, "unit_rate"] = round(PER_DIEM_CAP * f, 2)
                over = (lines.at[idx, "unit_rate"] - PER_DIEM_CAP) * lines.at[idx, "qty"]
                lines.at[idx, "amount"] = round(lines.at[idx, "qty"] * lines.at[idx, "unit_rate"], 2)
            delta = lines.at[idx, "amount"] - old_amt
            adjust_header(r.invoice_id, delta, r.line_type in ("SOV", "CO"))
            used_lines.add((r.invoice_id, int(r.line_no)))
            truth.append({"invoice_id": r.invoice_id, "line_no": int(r.line_no), "error_type": et,
                          "overbilled_amount": round(float(over), 2), "subtle": tiny})
        else:
            pool = [i for i in inv.invoice_id if i not in used_inv and (i, et) not in flagged_invoice_rules
                    and not str(i).endswith("-DUP")]
            if not pool:
                skipped.append(et)
                continue
            iid = pool[int(rng.integers(0, len(pool)))]
            used_inv.add(iid)
            i = inv.index[inv.invoice_id == iid][0]
            if et == "UNAPPROVED_CO":
                amt = round(max(1000.0, abs(inv.at[i, "gross_amount"]) * rng.uniform(0.02, 0.08)), 2)
                ln = int(lines.loc[lines.invoice_id == iid, "line_no"].max()) + 1
                new = {c: None for c in lines.columns}
                new.update(invoice_id=iid, line_no=ln, line_type="CO", ref_id=f"CO-INJ-{len(truth)+1:03d}",
                           description="Extra work (no approved change order)", unit="LS", qty=1.0,
                           unit_rate=amt, amount=amt, po_id=inv.at[i, "po_id"])
                lines = pd.concat([lines, pd.DataFrame([new])], ignore_index=True)
                adjust_header(iid, amt, True)
                truth.append({"invoice_id": iid, "line_no": ln, "error_type": et,
                              "overbilled_amount": amt, "subtle": False})
            elif et == "RETAINAGE_ERROR":
                old = inv.at[i, "retainage_withheld"]
                if old <= 1:
                    skipped.append(et)
                    continue
                inv.at[i, "retainage_withheld"] = round(old * 0.5, 2)
                inv.at[i, "net_amount"] = round(inv.at[i, "gross_amount"] - inv.at[i, "retainage_withheld"], 2)
                truth.append({"invoice_id": iid, "line_no": None, "error_type": et,
                              "overbilled_amount": round(old * 0.5, 2), "subtle": False})
            elif et == "DUPLICATE_INVOICE":
                dup_id = f"{iid}-DUP"
                dup_lines.append((iid, dup_id))  # built at the end from the final original
                truth.append({"invoice_id": dup_id, "line_no": None, "error_type": et,
                              "overbilled_amount": None, "subtle": False})

    # Duplicates copy the FINAL version of the original invoice (after any other planting)
    for orig, dup_id in dup_lines:
        dup = inv.loc[inv.invoice_id == orig].copy()
        dup["invoice_id"] = dup_id
        dup["invoice_number"] = dup["invoice_number"].astype(str) + "-R"
        dup["invoice_date"] = (pd.to_datetime(dup["invoice_date"]) + pd.Timedelta(days=10)).dt.strftime("%Y-%m-%d")
        dup_invoices.append(dup)
        for tr in truth:
            if tr["invoice_id"] == dup_id and tr["error_type"] == "DUPLICATE_INVOICE":
                tr["overbilled_amount"] = float(dup["net_amount"].iloc[0])
        dl = lines[lines.invoice_id == orig].copy()
        dl["invoice_id"] = dup_id
        lines = pd.concat([lines, dl], ignore_index=True)
        for tr in [x for x in truth if x["invoice_id"] == orig and x["line_no"] is not None]:
            truth.append({**tr, "invoice_id": dup_id})
    if dup_invoices:
        inv = pd.concat([inv] + dup_invoices, ignore_index=True)

    # Write modified copy
    mod = dict(t)
    mod["invoices"] = inv
    mod["invoice_lines"] = lines.drop(columns=["po_id"])
    for name, df in mod.items():
        df[[c for c, *_ in SCHEMA[name]] + [c for c in df.columns if c not in [c2 for c2, *_ in SCHEMA[name]]]] \
            .to_csv(out / f"{name}.csv", index=False)
    truth_df = pd.DataFrame(truth)
    truth_df.to_csv(out / "ground_truth_errors.csv", index=False)

    # Re-run and score
    ex, gates = run(tables=mod)
    det = keyset(ex)
    # baseline exceptions carried over (including onto duplicate copies) are not new
    expected_old = set(base_keys)
    for orig, dup_id in dup_lines:
        expected_old |= {(dup_id, l, r) for i, l, r in base_keys if i == orig}
    tru = set(zip(truth_df["invoice_id"], truth_df["line_no"].fillna(-1).astype(int), truth_df["error_type"]))
    caught, missed = tru & det, tru - det
    new_false = det - tru - expected_old

    rows = []
    for et in TYPES:
        sub = truth_df[truth_df.error_type == et] if len(truth_df) else truth_df
        k = {x for x in tru if x[2] == et}
        rows.append({"error_type": et, "planted": len(k), "caught": len(k & det), "missed": len(k - det),
                     "subtle_planted": int(sub["subtle"].sum()) if len(sub) else 0})
    rep = pd.DataFrame(rows)
    rep = rep[rep.planted > 0]
    tot_p, tot_c = rep.planted.sum(), rep.caught.sum()
    missed_df = truth_df[[(a, b, c) in missed for a, b, c in zip(truth_df.invoice_id,
                                                                  truth_df.line_no.fillna(-1).astype(int),
                                                                  truth_df.error_type)]]
    fa = ex[[k in new_false for k in zip(ex.invoice_id, ex.line_no.fillna(-1).astype(int), ex.rule)]]

    md = [
        "# Stress test report",
        "",
        f"Source data: `{args.data}` · planted: {tot_p} · seed {args.seed} · subtle share {args.subtle}",
        "",
        f"**Pre-existing exceptions in the source data:** {len(base_ex)} "
        f"(on {base_gates.gate.ne('PASS').sum()} of {len(base_gates)} invoices). "
        "These were left untouched. Review them by hand: each is either a real issue in your data or a mapping problem.",
        "",
        "## Planted vs caught",
        "",
        rep.to_markdown(index=False),
        "",
        f"**Recall on planted errors:** {100 * tot_c / max(tot_p, 1):.1f}% ({tot_c}/{tot_p})",
        f"**New false alarms caused by the engine:** {len(new_false)}",
        "",
        "## Missed (expected for subtle errors below tolerance)",
        "",
        missed_df.to_markdown(index=False) if len(missed_df) else "None.",
        "",
        "## New false alarms",
        "",
        fa.to_markdown(index=False) if len(fa) else "None.",
    ]
    if skipped:
        md += ["", f"_Skipped {len(skipped)} planned error(s) because the data had no suitable lines: "
                   f"{sorted(set(skipped))}_"]
    (out / "stress_test_report.md").write_text("\n".join(md))

    print(f"Baseline exceptions already in data: {len(base_ex)}")
    print(rep.to_string(index=False))
    print(f"\nRecall: {100 * tot_c / max(tot_p, 1):.1f}%   New false alarms: {len(new_false)}")
    if skipped:
        print(f"Skipped (no suitable lines): {sorted(set(skipped))}")
    print(f"\nModified data + answer key: {out}\nReport: {out / 'stress_test_report.md'}")


if __name__ == "__main__":
    main()
