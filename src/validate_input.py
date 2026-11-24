"""
Check a data folder BEFORE running the engine, so mapping mistakes don't turn into
silent misses or fake exceptions.

Usage:  python src/validate_input.py --data data/my_project

ERROR   = the engine can't run correctly; fix first
WARNING = the engine will run, but results may contain avoidable exceptions
"""
import argparse
import re
import sys
import pandas as pd

from schema import SCHEMA, REQUIRED_TABLES, read_folder

PERIOD = re.compile(r"^\d{4}-\d{2}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate(tables):
    issues = []

    def add(level, table, msg):
        issues.append({"level": level, "table": table, "message": msg})

    # 1. Files and columns
    for t in REQUIRED_TABLES:
        if t not in tables:
            add("ERROR", t, f"{t}.csv is missing (an empty file with just the header row is fine)")
            continue
        df = tables[t]
        for col, typ, req, _ in SCHEMA[t]:
            if col not in df.columns:
                add("ERROR" if req else "WARNING", t, f"column '{col}' is missing")
            elif typ in ("float", "int"):
                raw = df[col].dropna().astype(str).str.strip()
                raw = raw[raw != ""]
                bad = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce").isna().sum()
                if bad:
                    add("ERROR", t, f"column '{col}' has {bad} non-numeric value(s) (remove $ signs and text)")
                if raw.str.contains(",", regex=False).any():
                    add("ERROR", t, f"column '{col}' contains thousands separators (commas); save numbers without them")
    if any(i["level"] == "ERROR" for i in issues):
        return issues  # stop: later checks assume the structure is right

    po, pol = tables["purchase_orders"], tables["po_lines"]
    inv, lines = tables["invoices"], tables["invoice_lines"]
    grn, co, tm = tables["goods_receipts"], tables["change_orders"], tables["tm_preapprovals"]

    # 2. Unique keys
    for t, df, key in [("purchase_orders", po, ["po_id"]), ("po_lines", pol, ["po_id", "item_code"]),
                       ("invoices", inv, ["invoice_id"]), ("invoice_lines", lines, ["invoice_id", "line_no"]),
                       ("goods_receipts", grn, ["grn_id"]), ("change_orders", co, ["co_id"]),
                       ("tm_preapprovals", tm, ["approval_id"])]:
        d = df.duplicated(subset=key).sum()
        if d:
            add("ERROR", t, f"{d} duplicate key(s) on {key}")

    # 3. Links between tables
    def missing(child, col, parent, pcol):
        return set(child[col].dropna()) - set(parent[pcol].dropna())

    m = missing(inv, "po_id", po, "po_id")
    if m:
        add("ERROR", "invoices", f"po_id not found in purchase_orders: {sorted(m)[:5]}")
    m = missing(lines, "invoice_id", inv, "invoice_id")
    if m:
        add("ERROR", "invoice_lines", f"invoice_id not found in invoices: {sorted(m)[:5]}")
    m = missing(pol, "po_id", po, "po_id")
    if m:
        add("ERROR", "po_lines", f"po_id not found in purchase_orders: {sorted(m)[:5]}")
    no_lines = set(inv["invoice_id"]) - set(lines["invoice_id"])
    if no_lines:
        add("WARNING", "invoices", f"{len(no_lines)} invoice(s) have no lines, e.g. {sorted(no_lines)[:3]}")

    # 4. Allowed values and formats
    lt = set(lines["line_type"].dropna()) - {"SOV", "CO", "TM", "PER_DIEM"}
    if lt:
        add("ERROR", "invoice_lines", f"unknown line_type values {sorted(lt)} (allowed: SOV, CO, TM, PER_DIEM)")
    st = set(co["status"].dropna()) - {"APPROVED", "PENDING", "REJECTED"}
    if st:
        add("ERROR", "change_orders", f"unknown status values {sorted(st)} (allowed: APPROVED, PENDING, REJECTED)")
    for t, df, col, pat, fmt in [("invoices", inv, "period", PERIOD, "YYYY-MM"),
                                 ("invoices", inv, "invoice_date", DATE, "YYYY-MM-DD"),
                                 ("goods_receipts", grn, "period", PERIOD, "YYYY-MM"),
                                 ("tm_preapprovals", tm, "period", PERIOD, "YYYY-MM")]:
        bad = (~df[col].dropna().astype(str).str.match(pat)).sum()
        if bad:
            add("ERROR", t, f"{bad} value(s) in '{col}' are not in {fmt} format")
    rr = pd.to_numeric(po["retainage_rate"], errors="coerce")
    if (rr > 1).any():
        add("ERROR", "purchase_orders", "retainage_rate looks like a percent (e.g. 10); enter it as a fraction (0.10)")

    # 5. Line-level completeness (these would create avoidable exceptions)
    sov = lines[lines["line_type"] == "SOV"].merge(inv[["invoice_id", "po_id"]], on="invoice_id", how="left")
    is_ls = sov["unit"].astype(str).str.upper() == "LS"
    unit_sov, ls_sov = sov[~is_ls], sov[is_ls]
    n = unit_sov[["qty", "unit_rate"]].isna().any(axis=1).sum()
    if n:
        add("WARNING", "invoice_lines", f"{n} unit-price SOV line(s) missing qty or unit_rate")
    n = ls_sov[["pct_complete_prev", "pct_complete_cum"]].isna().any(axis=1).sum()
    if n:
        add("WARNING", "invoice_lines", f"{n} lump-sum line(s) missing pct_complete_prev / pct_complete_cum")
    keys = set(zip(pol["po_id"], pol["item_code"]))
    n = sum((p, r) not in keys for p, r in zip(sov["po_id"], sov["ref_id"]))
    if n:
        add("WARNING", "invoice_lines", f"{n} SOV line(s) reference an item_code not on that PO (rate check will skip them)")
    n = (~sov["grn_id"].isin(set(grn["grn_id"]))).sum()
    if n:
        add("WARNING", "invoice_lines", f"{n} SOV line(s) have no matching goods_receipts row (will be held as NO_RECEIPT)")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the input CSVs")
    args = ap.parse_args()
    issues = validate(read_folder(args.data))
    if not issues:
        print("OK: all checks passed. Ready to run the engine.")
        return 0
    for i in issues:
        print(f"{i['level']:8s} {i['table']:16s} {i['message']}")
    n_err = sum(i["level"] == "ERROR" for i in issues)
    print(f"\n{n_err} error(s), {len(issues) - n_err} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
