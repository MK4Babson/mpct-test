"""
US pay-application checks: AIA G702/G703-style cumulative ledger, stored materials, lien waivers, insurance,
and cost-plus / GMP cost checks. Deterministic rules; every finding is a recommendation for a person to decide.

  Ledger     G703 continuation sheet per bill: scheduled value, work from previous applications, this period,
             materials stored, total to date, % complete, balance to finish, retainage.
             G702 summary: contract sum to date, earned less retainage, previous certificates (from the OWNER's
             certified-payment ledger, not the contractor's figure), current payment due.
  Checks     PREVIOUS_CERT_MISMATCH       contractor's "less previous certificates" below the owner's ledger
             PENDING_CO_IN_CONTRACT_SUM   contract sum includes change orders not yet approved
             OVER_SCHEDULED_VALUE         an item billed past 100% of its scheduled value
             STORED_MATERIALS_UNSUPPORTED stored materials without bill of sale / insurance / inspection / receipt
             LIEN_WAIVER_MISSING          a required waiver (prime or subcontractor) not received
             INSURANCE_LAPSED             a required insurance certificate expired before the bill date
             INSURANCE_LIMIT_SHORT        certificate limit below the contract requirement
  GMP        BURDEN_OVER_CAP, MARKUP_ON_MARKUP, UNALLOWABLE_COST, CONTINGENCY_NOT_APPROVED,
             BUYOUT_SAVINGS_KEPT, GC_AHEAD_OF_TIME
"""
import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ["payments", "g702_submitted", "stored_materials", "lien_waivers", "insurance_certs", "gmp_terms", "gmp_costs", "payapp_truth"]
UNALLOWABLE = {"Bonus", "Legal", "Rework", "Fines", "Entertainment"}

LABEL = {
    "PREVIOUS_CERT_MISMATCH": "previous certificates understated",
    "PENDING_CO_IN_CONTRACT_SUM": "unapproved change orders in the contract sum",
    "OVER_SCHEDULED_VALUE": "lump sum billed past 100%",
    "QUANTITY_OVERRUN": "quantity beyond the contract quantity",
    "STORED_MATERIALS_UNSUPPORTED": "stored materials without required evidence",
    "LIEN_WAIVER_MISSING": "lien waiver not received",
    "INSURANCE_LAPSED": "insurance certificate expired",
    "INSURANCE_LIMIT_SHORT": "insurance limit below contract",
    "BURDEN_OVER_CAP": "labor burden above the agreed rate",
    "MARKUP_ON_MARKUP": "fee charged on the subcontractor's fee",
    "UNALLOWABLE_COST": "unallowable cost billed",
    "CONTINGENCY_NOT_APPROVED": "contingency used without owner approval",
    "BUYOUT_SAVINGS_KEPT": "buyout savings not passed to the owner",
    "GC_AHEAD_OF_TIME": "general conditions billed ahead of time",
}
ACTION = {
    "PREVIOUS_CERT_MISMATCH": "Restate the G702 with previous certificates from the owner's ledger; pay the recomputed amount.",
    "PENDING_CO_IN_CONTRACT_SUM": "Remove unapproved change orders from the contract sum until they are signed.",
    "OVER_SCHEDULED_VALUE": "Cap the item at its scheduled value; extra work needs an approved change order.",
    "QUANTITY_OVERRUN": "Confirm the measured quantity and whether the contract allows overruns at the unit rate.",
    "STORED_MATERIALS_UNSUPPORTED": "Hold the stored value until the missing document is supplied.",
    "LIEN_WAIVER_MISSING": "Hold the uncovered amount until the waiver arrives, or pay by joint check.",
    "INSURANCE_LAPSED": "No payment until a current certificate is on file.",
    "INSURANCE_LIMIT_SHORT": "Ask for an endorsed certificate at the contract limit.",
    "BURDEN_OVER_CAP": "Pay burden at the agreed cap and ask for the payroll burden calculation.",
    "MARKUP_ON_MARKUP": "Apply the prime's fee to the subcontractor's cost only, not to its fee.",
    "UNALLOWABLE_COST": "Remove the cost and its fee; it is not reimbursable under the contract.",
    "CONTINGENCY_NOT_APPROVED": "Reject the draw until the owner approves it in writing.",
    "BUYOUT_SAVINGS_KEPT": "Bill the subcontract at its award value and return the savings to contingency.",
    "GC_AHEAD_OF_TIME": "Pay general conditions in line with time elapsed; release the rest as the job progresses.",
}
SEVERITY = {"QUANTITY_OVERRUN": "REVIEW", "PENDING_CO_IN_CONTRACT_SUM": "REVIEW", "INSURANCE_LIMIT_SHORT": "REVIEW", "BUYOUT_SAVINGS_KEPT": "REVIEW",
            "GC_AHEAD_OF_TIME": "REVIEW"}


pa_cache = {}


def load(folder):
    folder = Path(folder)
    if not (folder / "g702_submitted.csv").exists():
        return None
    out = {}
    for f in FILES:
        p = folder / f"{f}.csv"
        if p.exists():
            out[f] = pd.read_csv(p, dtype={"invoice_id": str, "po_id": str, "prime_id": str, "period": str, "approval_ref": str,
                                           "cost_id": str, "package_id": str})
    pa_cache["payments"] = out.get("payments")
    return out


def _bills(t):
    inv = t["invoices"].copy()
    for c in ["gross_amount", "retainage_withheld", "net_amount"]:
        inv[c] = pd.to_numeric(inv[c], errors="coerce")
    inv = inv.sort_values(["po_id", "period", "invoice_id"])
    return inv[~inv.duplicated(["po_id", "invoice_number"], keep="first")]


def ledger(t):
    """Owner's certified-payment ledger: previous certificates per bill = sum paid on earlier bills of the same contract."""
    inv = _bills(t)
    pays = t.get("payments")
    if pays is None and pa_cache.get("payments") is not None:
        pays = pa_cache["payments"]
    paid = pd.to_numeric(pays.set_index("invoice_id").amount_paid, errors="coerce") if pays is not None else pd.Series(dtype=float)
    inv["certified"] = inv.invoice_id.map(paid).fillna(0)
    inv["previous_certified"] = inv.groupby("po_id").certified.transform(lambda s: s.shift(1, fill_value=0).cumsum())
    inv["application_no"] = inv.groupby("po_id").cumcount() + 1
    return inv


def _scheduled(t):
    pol = t["po_lines"].copy()
    pol["scheduled_value"] = pd.to_numeric(pol.scheduled_value, errors="coerce")
    return pol


def g703(t, pa, invoice_id):
    """Continuation sheet for one bill (one row per contract item billed on this contract so far)."""
    inv = ledger(t)
    r = inv[inv.invoice_id == invoice_id].iloc[0]
    lines = t["invoice_lines"].copy()
    lines["amount"] = pd.to_numeric(lines.amount, errors="coerce")
    lines = lines[lines.invoice_id.isin(inv[inv.po_id == r.po_id].invoice_id)].copy()
    lines.loc[lines.line_type == "TM", "ref_id"] = "T&M"
    lines.loc[lines.line_type == "PER_DIEM", "ref_id"] = "PER DIEM"
    order = inv[inv.po_id == r.po_id].set_index("invoice_id").application_no
    lines["app"] = lines.invoice_id.map(order)
    me = int(r.application_no)
    pol = _scheduled(t)
    pol = pol[pol.po_id == r.po_id]
    sv = pd.concat([pol.set_index("item_code").scheduled_value,
                    pol.dropna(subset=["co_id"]).set_index("co_id").scheduled_value]).groupby(level=0).sum()
    desc = pd.concat([pol.set_index("item_code").description, pol.dropna(subset=["co_id"]).set_index("co_id").description])
    desc = desc[~desc.index.duplicated()]
    desc["T&M"], desc["PER DIEM"] = "Time and materials (pre-approved hours; no retainage)", "Per diem (no retainage)"
    prev = lines[lines.app < me].groupby("ref_id").amount.sum()
    this = lines[lines.app == me].groupby("ref_id").amount.sum()
    items = sorted(set(prev.index) | set(this.index), key=lambda x: (x in ("T&M", "PER DIEM"), x))
    rows = []
    for it in items:
        c = float(sv.get(it, np.nan))
        d, e = float(prev.get(it, 0)), float(this.get(it, 0))
        rows.append({"item": it, "description": desc.get(it, ""), "scheduled_value": c, "from_previous": d, "this_period": e,
                     "stored": 0.0, "total_to_date": d + e, "pct": 100 * (d + e) / c if c else np.nan,
                     "balance_to_finish": c - d - e if c == c else np.nan})
    sm = pa.get("stored_materials", pd.DataFrame())
    for s in sm[sm.invoice_id == invoice_id].itertuples() if len(sm) else []:
        rows.append({"item": "STORED", "description": f"Materials stored: {s.description} ({s.location})", "scheduled_value": np.nan,
                     "from_previous": 0.0, "this_period": 0.0, "stored": float(s.value), "total_to_date": float(s.value),
                     "pct": np.nan, "balance_to_finish": np.nan})
    return pd.DataFrame(rows)


def g702(t, pa, invoice_id):
    """G702 summary, owner's computation next to the contractor's submitted figures."""
    inv = ledger(t)
    r = inv[inv.invoice_id == invoice_id].iloc[0]
    sheet = g703(t, pa, invoice_id)
    rate = float(t["purchase_orders"].set_index("po_id").retainage_rate.astype(float).get(r.po_id, 0.10))
    co = t["change_orders"].copy()
    co["co_amount"] = pd.to_numeric(co.co_amount, errors="coerce")
    approved = co[(co.po_id == r.po_id) & (co.status == "APPROVED") & (co.approval_period <= r.period)].co_amount.sum()
    pol = _scheduled(t)
    original = pol[(pol.po_id == r.po_id) & (pol.line_source == "CONTRACT")].scheduled_value.sum()
    work = sheet[sheet.item != "STORED"].total_to_date.sum()
    stored = sheet[sheet.item == "STORED"].stored.sum()
    retained_base = sheet[~sheet.item.isin(["T&M", "PER DIEM"])].total_to_date.sum()
    retain = rate * retained_base
    earned = work + stored - retain
    due = earned - r.previous_certified
    sub = pa["g702_submitted"].set_index("invoice_id").loc[invoice_id] if invoice_id in set(pa["g702_submitted"].invoice_id) else None
    lines = [
        ("1. Original contract sum", original, sub.original_contract_sum if sub is not None else np.nan),
        ("2. Net change by change orders", approved, sub.net_change_by_cos if sub is not None else np.nan),
        ("3. Contract sum to date", original + approved,
         (sub.original_contract_sum + sub.net_change_by_cos) if sub is not None else np.nan),
        ("4. Total completed and stored to date", work + stored, np.nan),
        (f"5. Retainage ({rate:.0%} on contract work and stored materials)", retain, np.nan),
        ("6. Total earned less retainage", earned, np.nan),
        ("7. Less previous certificates for payment", r.previous_certified, sub.less_previous_certificates if sub is not None else np.nan),
        ("8. Current payment due", due, (earned - sub.less_previous_certificates) if sub is not None else np.nan),
        ("9. Balance to finish, including retainage", original + approved - earned, np.nan),
    ]
    df = pd.DataFrame(lines, columns=["line", "owner", "contractor"])
    df["difference"] = (df.contractor - df.owner).round(2)
    df.loc[df.difference.abs() < 0.5, "difference"] = 0.0
    return df, r


def _row(invoice_id, rule, amount, detail, prime_id=None, period=None, ref=None):
    return {"invoice_id": invoice_id, "line_no": ref, "rule": rule, "severity": SEVERITY.get(rule, "HOLD"), "detail": detail,
            "amount_at_risk": round(float(amount), 2), "prime_id": prime_id, "period": period, "source": "Pay application"}


def checks(t, pa):
    """All pay-application and compliance findings, one row each."""
    if pa is None:
        return pd.DataFrame(columns=["invoice_id", "line_no", "rule", "severity", "detail", "amount_at_risk", "prime_id", "period",
                                     "source"])
    inv = ledger(t)
    meta = inv.set_index("invoice_id")
    out = []
    g = pa["g702_submitted"].merge(inv[["invoice_id", "previous_certified", "prime_id", "period"]], on="invoice_id", suffixes=("", "_l"))
    co = t["change_orders"].copy()
    co["co_amount"] = pd.to_numeric(co.co_amount, errors="coerce")
    for r in g.itertuples():
        gap = r.previous_certified - r.less_previous_certificates
        if gap > 1:
            out.append(_row(r.invoice_id, "PREVIOUS_CERT_MISMATCH", gap,
                            f"G702 shows ${r.less_previous_certificates:,.0f} previously certified; the owner's ledger shows "
                            f"${r.previous_certified:,.0f}. The current payment due is overstated by ${gap:,.0f}.", r.prime_id, r.period))
        appr = co[(co.po_id == r.po_id) & (co.status == "APPROVED") & (co.approval_period <= r.period)].co_amount.sum()
        extra = r.net_change_by_cos - appr
        if extra > 1:
            out.append(_row(r.invoice_id, "PENDING_CO_IN_CONTRACT_SUM", extra,
                            f"Net change by change orders ${r.net_change_by_cos:,.0f}; approved on file ${appr:,.0f}.",
                            r.prime_id, r.period))
    # over scheduled value (cumulative per item)
    lines = t["invoice_lines"].copy()
    lines["amount"] = pd.to_numeric(lines.amount, errors="coerce")
    lines = lines[lines.invoice_id.isin(inv.invoice_id) & lines.line_type.isin(["SOV", "CO"])].merge(
        inv[["invoice_id", "po_id", "application_no", "prime_id", "period"]], on="invoice_id")
    lines = lines.sort_values(["po_id", "ref_id", "application_no"])
    lines["cum"] = lines.groupby(["po_id", "ref_id"]).amount.cumsum()
    pol = _scheduled(t)
    sv = pd.concat([pol.set_index(["po_id", "item_code"]).scheduled_value,
                    pol.dropna(subset=["co_id"]).set_index(["po_id", "co_id"]).scheduled_value]).groupby(level=[0, 1]).sum()
    lines["sv"] = [sv.get((a, b), np.nan) for a, b in zip(lines.po_id, lines.ref_id)]
    over = lines[lines.cum > lines.sv * 1.005]
    over = over.groupby(["po_id", "ref_id"]).head(1)
    for r in over.itertuples():
        if r.unit == "LS":
            out.append(_row(r.invoice_id, "OVER_SCHEDULED_VALUE", r.cum - r.sv,
                            f"Lump-sum item {r.ref_id} billed ${r.cum:,.0f} to date against its scheduled value of ${r.sv:,.0f} "
                            f"({100 * r.cum / r.sv:.0f}%).", r.prime_id, r.period, r.line_no))
        else:
            out.append(_row(r.invoice_id, "QUANTITY_OVERRUN", r.cum - r.sv,
                            f"Unit-price item {r.ref_id} billed ${r.cum:,.0f} to date against ${r.sv:,.0f} at the contract quantity. "
                            "Overruns can be legitimate under a unit-price contract; check the variation clause.",
                            r.prime_id, r.period, r.line_no))
    # stored materials
    for r in pa.get("stored_materials", pd.DataFrame()).itertuples():
        need = ["bill_of_sale", "insurance_certificate", "owner_inspection"] if r.location.startswith("Off") else ["delivery_receipt"]
        miss = [n.replace("_", " ") for n in need if getattr(r, n) != "Y"]
        if miss:
            m = meta.loc[r.invoice_id]
            out.append(_row(r.invoice_id, "STORED_MATERIALS_UNSUPPORTED", r.value,
                            f"{r.description}, {r.location}, ${r.value:,.0f}: missing {', '.join(miss)}.", m.prime_id, m.period))
    # lien waivers
    for r in pa.get("lien_waivers", pd.DataFrame()).itertuples():
        if r.received != "Y" and r.invoice_id in meta.index:
            m = meta.loc[r.invoice_id]
            out.append(_row(r.invoice_id, "LIEN_WAIVER_MISSING", r.amount_covered,
                            f"{r.waiver_type} waiver from {r.party} not received (covers ${r.amount_covered:,.0f}).", m.prime_id, m.period))
    # insurance
    coi = pa.get("insurance_certs", pd.DataFrame())
    for r in inv.itertuples():
        c = coi[coi.prime_id == r.prime_id] if len(coi) else coi
        lapsed = c[c.expires < r.invoice_date] if len(c) else c
        for x in lapsed.itertuples() if len(lapsed) else []:
            out.append(_row(r.invoice_id, "INSURANCE_LAPSED", 0, f"{x.coverage} certificate expired {x.expires}; bill dated {r.invoice_date}.",
                            r.prime_id, r.period))
    for x in coi[coi.limit_on_certificate < coi.limit_required].itertuples() if len(coi) else []:
        last = inv[inv.prime_id == x.prime_id].sort_values("period").iloc[-1]
        out.append(_row(last.invoice_id, "INSURANCE_LIMIT_SHORT", 0, f"{x.coverage}: certificate shows ${x.limit_on_certificate:,.0f}, "
                        f"contract requires ${x.limit_required:,.0f}.", x.prime_id, last.period))
    return pd.DataFrame(out)


def gmp_checks(pa):
    """Cost-plus / GMP findings on the general conditions and reimbursables packages."""
    if pa is None or "gmp_costs" not in pa:
        return pd.DataFrame()
    gm, terms = pa["gmp_costs"].copy(), pa["gmp_terms"].set_index("prime_id")
    out = []
    for r in gm.itertuples():
        t = terms.loc[r.prime_id]
        fee = t.fee_pct / 100
        if r.category == "Labor" and r.labor_burden_pct > t.labor_burden_cap_pct:
            amt = r.base_cost * (r.labor_burden_pct - t.labor_burden_cap_pct) / 100 * (1 + fee)
            out.append(_row(r.cost_id, "BURDEN_OVER_CAP", amt, f"Burden {r.labor_burden_pct:.0f}% on ${r.base_cost:,.0f} of labor; "
                            f"agreed cap {t.labor_burden_cap_pct:.0f}%.", r.prime_id, r.period))
        if r.category == "Subcontract" and r.sub_fee_included == "Y" and r.sub_fee_amount > 0:
            out.append(_row(r.cost_id, "MARKUP_ON_MARKUP", r.sub_fee_amount * fee, f"Prime fee {t.fee_pct:.0f}% charged on a subcontract "
                            f"amount that already includes the sub's ${r.sub_fee_amount:,.0f} fee.", r.prime_id, r.period))
        if r.category in UNALLOWABLE:
            out.append(_row(r.cost_id, "UNALLOWABLE_COST", r.billed_amount, f"{r.category}: {r.description}, "
                            f"${r.billed_amount:,.0f} with fee.", r.prime_id, r.period))
        if r.category == "Contingency draw" and not (isinstance(r.approval_ref, str) and r.approval_ref.strip()):
            out.append(_row(r.cost_id, "CONTINGENCY_NOT_APPROVED", r.billed_amount, f"{r.description}: ${r.billed_amount:,.0f} drawn "
                            "with no owner approval reference.", r.prime_id, r.period))
        if r.category == "Subcontract buyout" and r.budget_value > r.award_value and r.base_cost >= r.budget_value:
            amt = (r.budget_value - r.award_value) * (1 + fee)
            out.append(_row(r.cost_id, "BUYOUT_SAVINGS_KEPT", amt, f"{r.description}: awarded at ${r.award_value:,.0f}, billed at the "
                            f"budget ${r.budget_value:,.0f}.", r.prime_id, r.period))
    # general conditions pacing, latest period per package
    gc = gm[gm.category == "General conditions"].sort_values("period")
    for p, g in gc.groupby("prime_id"):
        t = terms.loc[p]
        g = g.assign(cum=g.base_cost.cumsum())
        start = pd.Period(t.start_period, "M")
        last = g.iloc[-1]
        months = (pd.Period(last.period, "M") - start).n + 1
        share, elapsed = last.cum / t.gc_budget, months / t.contract_months
        if share - elapsed > 0.05:
            amt = (share - elapsed) * t.gc_budget
            out.append(_row(t.package_id, "GC_AHEAD_OF_TIME", amt, f"{share:.0%} of the general-conditions budget billed after "
                            f"{months} of {t.contract_months} months ({elapsed:.0%} of the time).", p, last.period))
    return pd.DataFrame(out)


def accuracy(found, pa):
    """Compare findings with the planted answer key, by check."""
    if pa is None or "payapp_truth" not in pa:
        return pd.DataFrame()
    truth = pa["payapp_truth"]
    f = set(zip(found.invoice_id.astype(str), found.rule))
    tr = set(zip(truth.invoice_id.astype(str), truth.check))
    rows = []
    for rule in sorted({r for _, r in tr | f}):
        d, t_ = {x for x in f if x[1] == rule}, {x for x in tr if x[1] == rule}
        rows.append({"check": LABEL.get(rule, rule), "planted": len(t_), "caught": len(d & t_), "missed": len(t_ - d),
                     "found_not_planted": len(d - t_)})
    return pd.DataFrame(rows)


def notice_docx(bill, items, owner_name="[Owner]", contractor="[Contractor]", signer="[Name, title]"):
    """Draft written notice of payment withholding (Word). Items: rows with rule, detail, amount."""
    from docx import Document
    from docx.shared import Pt

    d = Document()
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(10.5)
    d.add_heading("Notice of payment withholding", level=1)
    p = d.add_paragraph()
    p.add_run("DRAFT for review by the contract administrator and counsel before sending.").bold = True
    d.add_paragraph(f"Date: {date.today():%d %B %Y}\nTo: {contractor}\nFrom: {owner_name}\n"
                    f"Re: Payment application {bill['invoice_number']} ({bill['period']}), contract {bill['po_id']}")
    total = float(sum(i["amount"] for i in items))
    req = float(bill.get("net_amount", np.nan))
    d.add_paragraph(
        f"We have reviewed the payment application above. We will pay the undisputed portion of ${max(req - total, 0):,.2f} "
        f"within the contract's payment terms. We are withholding ${total:,.2f} for the reasons below. Each item will be released "
        "when the document or correction listed is received.")
    tb = d.add_table(rows=1, cols=4)
    tb.style = "Light Grid Accent 1"
    for c, h in zip(tb.rows[0].cells, ["Item", "Reason", "Contract basis", "Amount withheld"]):
        c.text = h
    for i in items:
        row = tb.add_row().cells
        row[0].text = str(i.get("ref", ""))
        row[1].text = f"{LABEL.get(i['rule'], i['rule'])}. {i['detail']}"
        row[2].text = "[Contract section ___]"
        row[3].text = f"${i['amount']:,.2f}"
    d.add_paragraph()
    d.add_paragraph("To release the withheld amounts, please provide: " + "; ".join(
        sorted({ACTION.get(i["rule"], "the supporting documents") for i in items})))
    d.add_paragraph("This notice is given under the payment provisions of the contract and applicable prompt-payment law. "
                    "Check the state's notice deadline and required content before sending.")
    d.add_paragraph(f"\n{signer}\n{owner_name}")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
