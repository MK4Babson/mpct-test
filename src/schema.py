"""
Input schema for the engine, plus a loader that reads any data folder in this format.

Every table is a CSV file named <table>.csv. Columns marked required=True must exist
(values may still be blank where the data dictionary says so). Optional columns are
added as blanks if missing. Types: str, float, int.
"""
from pathlib import Path
import pandas as pd

# table -> list of (column, type, required, meaning)
SCHEMA = {
    "purchase_orders": [
        ("po_id", "str", True, "Unique contract / PO id"),
        ("project_id", "str", True, "Building or project the PO belongs to"),
        ("prime_id", "str", True, "Prime contractor id"),
        ("retainage_rate", "float", True, "Retainage as a fraction, e.g. 0.10 for 10%"),
        ("contract_type", "str", False, "e.g. Unit price, Lump sum, GMP"),
    ],
    "po_lines": [
        ("po_id", "str", True, "PO this line belongs to"),
        ("item_code", "str", True, "Pay item / SOV code (for approved change orders, the CO id)"),
        ("line_source", "str", True, "CONTRACT or CHANGE_ORDER"),
        ("description", "str", True, "Item description"),
        ("unit", "str", True, "LF, CY, EA, TON, LS, ..."),
        ("contract_qty", "float", True, "Contract quantity (1 for lump sum)"),
        ("contract_unit_rate", "float", True, "Contract unit price (scheduled value for lump sum)"),
        ("scheduled_value", "float", True, "contract_qty x contract_unit_rate"),
        ("co_id", "str", False, "Change order id when line_source = CHANGE_ORDER"),
    ],
    "change_orders": [
        ("co_id", "str", True, "Change order id"),
        ("po_id", "str", True, "PO it modifies"),
        ("status", "str", True, "APPROVED, PENDING or REJECTED"),
        ("approval_period", "str", True, "YYYY-MM it was approved (blank if not approved)"),
        ("co_qty", "float", False, "Quantity"),
        ("co_unit_rate", "float", False, "Unit rate"),
        ("co_amount", "float", False, "Total value"),
        ("description", "str", False, "What the change is"),
    ],
    "tm_preapprovals": [
        ("approval_id", "str", True, "Pre-approval id referenced by T&M invoice lines"),
        ("po_id", "str", True, "PO"),
        ("period", "str", True, "YYYY-MM"),
        ("craft", "str", False, "Labor craft"),
        ("approved_hours", "float", True, "Hours approved"),
        ("approved_rate", "float", False, "Approved hourly rate"),
    ],
    "goods_receipts": [
        ("grn_id", "str", True, "Receipt / field verification id referenced by invoice lines"),
        ("po_id", "str", True, "PO"),
        ("item_code", "str", True, "Pay item verified"),
        ("period", "str", True, "YYYY-MM"),
        ("verified_qty", "float", True, "Quantity verified this period (blank for lump sum)"),
        ("verified_pct_complete_cum", "float", True, "Verified cumulative % complete (lump sum only)"),
    ],
    "invoices": [
        ("invoice_id", "str", True, "Unique system id for the invoice"),
        ("invoice_number", "str", True, "Contractor's invoice / pay app number"),
        ("po_id", "str", True, "PO billed"),
        ("project_id", "str", True, "Project"),
        ("prime_id", "str", True, "Prime contractor"),
        ("period", "str", True, "Billing period YYYY-MM"),
        ("invoice_date", "str", True, "YYYY-MM-DD"),
        ("gross_amount", "float", True, "Sum of line amounts"),
        ("retainage_withheld", "float", True, "Retainage withheld on this invoice"),
        ("net_amount", "float", True, "gross - retainage"),
    ],
    "invoice_lines": [
        ("invoice_id", "str", True, "Invoice"),
        ("line_no", "int", True, "Line number within the invoice"),
        ("line_type", "str", True, "SOV, CO, TM or PER_DIEM"),
        ("ref_id", "str", True, "SOV: item_code | CO: co_id | TM: approval_id | PER_DIEM: any"),
        ("description", "str", True, "Line description"),
        ("unit", "str", True, "Unit (LS for lump sum)"),
        ("qty", "float", True, "Quantity billed this period (blank for lump sum)"),
        ("unit_rate", "float", True, "Unit rate billed (blank for lump sum)"),
        ("amount", "float", True, "Amount billed"),
        ("pct_complete_prev", "float", True, "Lump sum: % complete billed before this invoice"),
        ("pct_complete_cum", "float", True, "Lump sum: cumulative % complete billed"),
        ("grn_id", "str", True, "Receipt id this SOV line should be matched to"),
    ],
    # Optional lookup tables (used for names only)
    "primes": [("prime_id", "str", True, "Prime id"), ("prime_name", "str", True, "Prime name")],
    "projects": [("project_id", "str", True, "Project id")],
}
REQUIRED_TABLES = [t for t in SCHEMA if t not in ("primes", "projects")]
SQL_TYPES = {"str": "VARCHAR", "float": "DOUBLE", "int": "BIGINT"}


def coerce(df, table):
    """Return df with schema columns in the right types (optional columns added if missing)."""
    df = df.copy()
    for col, typ, _req, _m in SCHEMA[table]:
        if col not in df.columns:
            df[col] = None
        if typ == "str":
            df[col] = df[col].astype("object").where(df[col].notna(), None)
            df[col] = df[col].map(lambda v: None if v is None else str(v).strip())
        elif typ == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def read_folder(data_dir):
    """Read every <table>.csv found in data_dir (strings kept as text, e.g. '0105' stays '0105')."""
    data_dir = Path(data_dir)
    tables = {}
    for t, cols in SCHEMA.items():
        f = data_dir / f"{t}.csv"
        if f.exists():
            str_cols = {c: str for c, typ, _, _ in cols if typ == "str"}
            tables[t] = pd.read_csv(f, dtype=str_cols)
    return tables


def write_templates(folder):
    """Write header-only CSV templates for every table."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for t, cols in SCHEMA.items():
        pd.DataFrame(columns=[c for c, *_ in cols]).to_csv(folder / f"{t}.csv", index=False)
