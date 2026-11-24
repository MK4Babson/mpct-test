"""
Multi-Prime Control Tower - bill-checking API.

The same checking engine as the Streamlit app (sql/engine.sql), exposed so other software can use it:
a contractor portal, an accounts-payable system, a spreadsheet macro or a Looker dashboard.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for the interactive documentation.

All bundled data is synthetic (fictional contractors, sites and amounts).
"""
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import PER_DIEM_CAP, TOLERANCES  # noqa: E402
from insights import RULE_ACTION, RULE_LABEL  # noqa: E402
from run_engine import run, scorecard  # noqa: E402
from schema import REQUIRED_TABLES, SCHEMA, read_folder  # noqa: E402

PROGRAMS = {
    "west-texas": {"name": "West Texas data-center program", "folder": ROOT / "data" / "raw",
                   "currency": "USD"},
    "north-koel": {"name": "North Koel canal program", "folder": ROOT / "data" / "north_koel",
                   "currency": "INR"},
}

app = FastAPI(
    title="Multi-Prime Control Tower API",
    version="1.0.0",
    description=(
        "Checks construction bills against the contract, approved change orders and what the site team "
        "actually verified. Every bill comes back as **PASS**, **REVIEW** or **HOLD**, with the reason in plain "
        "English, the dollars in question and the next step.\n\n"
        "**All bundled data is synthetic.** Contractors, sites and amounts are fictional."
    ),
)


# ---------- security (optional) ----------
def require_key(x_api_key: Optional[str] = Header(default=None)):
    """If MPCT_API_KEY is set on the server, POST endpoints need the header X-API-Key."""
    expected = os.environ.get("MPCT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or wrong X-API-Key header.")


# ---------- helpers ----------
def _program(program_id):
    if program_id not in PROGRAMS:
        raise HTTPException(404, f"Unknown program '{program_id}'. Try one of: {', '.join(PROGRAMS)}.")
    return PROGRAMS[program_id]


@lru_cache(maxsize=None)
def _tables(program_id):
    return read_folder(_program(program_id)["folder"])


@lru_cache(maxsize=None)
def _results(program_id):
    tables = _tables(program_id)
    ex, gates = run(tables=tables)
    return ex, gates


def _clean(df):
    """DataFrame -> JSON-safe list of dicts (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _reasons(ex):
    out = []
    for r in _clean(ex):
        out.append({
            "line_no": r["line_no"],
            "rule": r["rule"],
            "severity": r["severity"],
            "what_is_wrong": RULE_LABEL.get(r["rule"], r["rule"]),
            "detail": r["detail"],
            "amount_in_question": round(float(r["amount_at_risk"] or 0), 2),
            "next_step": RULE_ACTION.get(r["rule"], ""),
        })
    return out


def _verdict(gate_row, ex):
    g = gate_row
    return {
        "invoice_id": g["invoice_id"],
        "invoice_number": g["invoice_number"],
        "project_id": g["project_id"],
        "contractor_id": g["prime_id"],
        "period": g["period"],
        "gross_amount": round(float(g["gross_amount"]), 2),
        "decision": g["gate"],
        "amount_in_question": round(float(g["amount_at_risk"] or 0), 2),
        "safe_to_pay_now": round(float(g["net_amount"]) - float(g["amount_at_risk"] or 0), 2)
        if g["gate"] != "PASS" else round(float(g["net_amount"]), 2),
        "reasons": _reasons(ex),
    }


# ---------- request models ----------
class BillLine(BaseModel):
    line_no: int
    line_type: Literal["SOV", "CO", "TM", "PER_DIEM"] = Field(
        description="SOV = contract item, CO = change order, TM = time & materials, PER_DIEM = per diem")
    ref_id: str = Field(description="SOV: item code | CO: change-order id | TM: pre-approval id | PER_DIEM: any")
    description: str
    unit: str
    qty: Optional[float] = None
    unit_rate: Optional[float] = None
    amount: float
    pct_complete_prev: Optional[float] = None
    pct_complete_cum: Optional[float] = None
    grn_id: Optional[str] = Field(default=None, description="Field-verification id this line should match")


class FieldVerification(BaseModel):
    grn_id: str
    item_code: str
    verified_qty: Optional[float] = None
    verified_pct_complete_cum: Optional[float] = None


class Bill(BaseModel):
    invoice_id: str
    invoice_number: str
    po_id: str
    period: str = Field(description="Billing month, YYYY-MM")
    invoice_date: str = Field(description="YYYY-MM-DD")
    gross_amount: float
    retainage_withheld: float
    net_amount: float
    lines: list[BillLine]
    field_verifications: list[FieldVerification] = Field(
        default_factory=list,
        description="What the site team verified for this bill. Optional if already on file in the program.")


EXAMPLE_BILL = {
    "invoice_id": "API-0001",
    "invoice_number": "CC2-009",
    "po_id": "PO-WTX-C2-C",
    "period": "2026-09",
    "invoice_date": "2026-09-29",
    "gross_amount": 4266500.00,
    "retainage_withheld": 426650.00,
    "net_amount": 3839850.00,
    "lines": [
        {"line_no": 1, "line_type": "SOV", "ref_id": "C-301", "description": "Mass excavation", "unit": "CY",
         "qty": 50000, "unit_rate": 9.33, "amount": 466500.00, "grn_id": "GRN-API-C301"},
        {"line_no": 2, "line_type": "SOV", "ref_id": "C-302", "description": "Structural concrete - foundations",
         "unit": "CY", "qty": 8000, "unit_rate": 475.00, "amount": 3800000.00, "grn_id": "GRN-API-C302"},
    ],
    "field_verifications": [
        {"grn_id": "GRN-API-C301", "item_code": "C-301", "verified_qty": 48000},
        {"grn_id": "GRN-API-C302", "item_code": "C-302", "verified_qty": 8000},
    ],
}


class Tolerances(BaseModel):
    qty_review_pct: float = TOLERANCES["qty_review_pct"]
    qty_hold_pct: float = TOLERANCES["qty_hold_pct"]
    rate_round_usd: float = TOLERANCES["rate_round_usd"]
    pct_review_pts: float = TOLERANCES["pct_review_pts"]
    pct_hold_pts: float = TOLERANCES["pct_hold_pts"]
    tm_review_pct: float = TOLERANCES["tm_review_pct"]
    tm_hold_pct: float = TOLERANCES["tm_hold_pct"]
    money_tol: float = TOLERANCES["money_tol"]


class FullDataset(BaseModel):
    """Bring your own data: one list of rows per table, same columns as the CSV templates."""
    tables: dict[str, list[dict]]
    tolerances: Optional[Tolerances] = None
    per_diem_cap: float = PER_DIEM_CAP


# ---------- endpoints: information ----------
@app.get("/", tags=["About"])
def home():
    return {
        "name": "Multi-Prime Control Tower API",
        "what_it_does": "Checks construction bills against contract, approvals and verified work; "
                        "returns PASS / REVIEW / HOLD with reasons and dollars in question.",
        "docs": "/docs",
        "programs": list(PROGRAMS),
        "data_note": "All bundled data is synthetic.",
    }


@app.get("/health", tags=["About"])
def health():
    return {"status": "ok"}


@app.get("/rules", tags=["About"], summary="The checks the engine runs, in plain English")
def rules():
    return {
        "rules": [{"rule": k, "what_it_catches": RULE_LABEL[k], "next_step": RULE_ACTION.get(k, "")}
                  for k in RULE_LABEL],
        "default_tolerances": TOLERANCES,
        "per_diem_cap_per_worker_day": PER_DIEM_CAP,
    }


@app.get("/programs", tags=["Programs"])
def programs():
    return [{"program_id": k, "name": v["name"], "currency": v["currency"]} for k, v in PROGRAMS.items()]


@app.get("/programs/{program_id}/summary", tags=["Programs"], summary="Totals across every bill")
def summary(program_id: str):
    _program(program_id)
    _, g = _results(program_id)
    counts = g.gate.value_counts()
    return {
        "program_id": program_id,
        "bills": int(len(g)),
        "billed_gross": round(float(g.gross_amount.sum()), 2),
        "amount_in_question": round(float(g.amount_at_risk.sum()), 2),
        "pass": int(counts.get("PASS", 0)),
        "review": int(counts.get("REVIEW", 0)),
        "hold": int(counts.get("HOLD", 0)),
        "first_pass_rate_pct": round(100 * counts.get("PASS", 0) / max(len(g), 1), 1),
        "periods": sorted(g.period.dropna().unique().tolist()),
    }


@app.get("/programs/{program_id}/bills", tags=["Programs"], summary="List bills, with filters")
def bills(program_id: str,
          decision: Optional[Literal["PASS", "REVIEW", "HOLD"]] = None,
          contractor_id: Optional[str] = None,
          project_id: Optional[str] = None,
          period: Optional[str] = Query(default=None, description="YYYY-MM"),
          limit: int = Query(default=200, le=1000)):
    _program(program_id)
    _, g = _results(program_id)
    if decision:
        g = g[g.gate == decision]
    if contractor_id:
        g = g[g.prime_id == contractor_id]
    if project_id:
        g = g[g.project_id == project_id]
    if period:
        g = g[g.period == period]
    cols = ["invoice_id", "invoice_number", "project_id", "prime_id", "period", "gross_amount", "gate",
            "amount_at_risk", "rules_triggered"]
    out = g[cols].rename(columns={"prime_id": "contractor_id", "gate": "decision",
                                  "amount_at_risk": "amount_in_question"}).head(limit)
    return {"count": int(len(g)), "bills": _clean(out)}


@app.get("/programs/{program_id}/bills/{invoice_id}", tags=["Programs"],
         summary="One bill: decision, reasons and next steps")
def bill(program_id: str, invoice_id: str):
    _program(program_id)
    ex, g = _results(program_id)
    row = g[g.invoice_id == invoice_id]
    if row.empty:
        raise HTTPException(404, f"No bill '{invoice_id}' in {program_id}.")
    return _verdict(row.iloc[0], ex[ex.invoice_id == invoice_id])


@app.get("/programs/{program_id}/contractors", tags=["Programs"], summary="Contractor scorecard")
def contractors(program_id: str):
    _program(program_id)
    ex, g = _results(program_id)
    sc = scorecard(g, ex, _tables(program_id).get("primes"))
    return _clean(sc.rename(columns={"prime_id": "contractor_id", "prime_name": "contractor_name"}))


# ---------- endpoints: checking ----------
@app.post("/programs/{program_id}/check-bill", tags=["Check bills"], dependencies=[Depends(require_key)],
          summary="Check one new bill against a program's contracts")
def check_bill(program_id: str, bill: Annotated[Bill, Body(examples=[EXAMPLE_BILL])]):
    """Send a new bill (and, optionally, what the site team verified). The API checks it against the program's
    contracts, change orders, T&M approvals and earlier bills (so re-sent bills are caught as duplicates).
    Nothing is saved."""
    _program(program_id)
    base = _tables(program_id)
    po = base["purchase_orders"]
    match = po[po.po_id == bill.po_id]
    if match.empty:
        raise HTTPException(422, f"PO '{bill.po_id}' is not in {program_id}. Check the contract number.")
    if bill.invoice_id in set(base["invoices"].invoice_id):
        raise HTTPException(409, f"invoice_id '{bill.invoice_id}' already exists; use a new id.")
    p = match.iloc[0]
    tables = {k: v.copy() for k, v in base.items()}
    head = bill.model_dump(exclude={"lines", "field_verifications"})
    head.update(project_id=p.project_id, prime_id=p.prime_id)
    tables["invoices"] = pd.concat([tables["invoices"], pd.DataFrame([head])], ignore_index=True)
    lines = pd.DataFrame([{**ln.model_dump(), "invoice_id": bill.invoice_id} for ln in bill.lines])
    tables["invoice_lines"] = pd.concat([tables["invoice_lines"], lines], ignore_index=True)
    if bill.field_verifications:
        gr = pd.DataFrame([{**v.model_dump(), "po_id": bill.po_id, "period": bill.period}
                           for v in bill.field_verifications])
        tables["goods_receipts"] = pd.concat([tables["goods_receipts"], gr], ignore_index=True)
    ex, g = run(tables=tables)
    row = g[g.invoice_id == bill.invoice_id].iloc[0]
    return _verdict(row, ex[ex.invoice_id == bill.invoice_id])


@app.post("/check", tags=["Check bills"], dependencies=[Depends(require_key)],
          summary="Bring your own data: check every bill in a full dataset")
def check_dataset(data: FullDataset):
    """Send all tables (purchase_orders, po_lines, change_orders, tm_preapprovals, goods_receipts, invoices,
    invoice_lines; primes and projects optional) as lists of rows. Column names match the CSV templates
    returned by GET /schema. Optionally loosen or tighten the tolerances."""
    missing = [t for t in REQUIRED_TABLES if t not in data.tables]
    if missing:
        raise HTTPException(422, f"Missing tables: {', '.join(missing)}. See GET /schema.")
    tables = {k: pd.DataFrame(v) for k, v in data.tables.items() if k in SCHEMA}
    tol = data.tolerances.model_dump() if data.tolerances else None
    try:
        ex, g = run(tolerances=tol, tables=tables, per_diem_cap=data.per_diem_cap)
    except Exception as e:  # bad column types, etc.
        raise HTTPException(422, f"Could not check the data: {e}")
    counts = g.gate.value_counts()
    return {
        "summary": {"bills": int(len(g)), "pass": int(counts.get("PASS", 0)),
                    "review": int(counts.get("REVIEW", 0)), "hold": int(counts.get("HOLD", 0)),
                    "amount_in_question": round(float(g.amount_at_risk.sum()), 2)},
        "bills": [_verdict(r, ex[ex.invoice_id == r["invoice_id"]]) for r in _clean(g)],
    }


@app.get("/schema", tags=["Check bills"], summary="Tables and columns the /check endpoint expects")
def schema():
    return {t: [{"column": c, "type": typ, "required": req, "meaning": m} for c, typ, req, m in cols]
            for t, cols in SCHEMA.items()}
