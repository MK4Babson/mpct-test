"""
Build the US pay-application layer for the West Texas program (synthetic, with an answer key).

Adds what an owner's team actually receives with each monthly AIA-style pay application:
  g702_submitted.csv     the contractor's G702 summary as submitted (contract sum, changes, previous certificates)
  stored_materials.csv   materials billed as stored (on site or off site) and the evidence supplied
  lien_waivers.csv       waivers expected with the bill (prime + subcontractors) and whether received
  insurance_certs.csv    certificates of insurance per prime contractor with expiry dates
  gmp_terms.csv          cost-plus / GMP terms for each prime's general conditions & reimbursables package
  gmp_costs.csv          monthly cost-plus cost detail billed under that package
  payapp_truth.csv       the planted errors (answer key)

Everything is fictional. Run: python src/build_payapp.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
rng = np.random.default_rng(42)


def main():
    inv = pd.read_csv(RAW / "invoices.csv", dtype=str)
    for c in ["gross_amount", "retainage_withheld", "net_amount"]:
        inv[c] = inv[c].astype(float)
    po = pd.read_csv(RAW / "purchase_orders.csv", dtype=str)
    pol = pd.read_csv(RAW / "po_lines.csv", dtype={"po_id": str, "item_code": str})
    co = pd.read_csv(RAW / "change_orders.csv", dtype=str)
    co["co_amount"] = co.co_amount.astype(float)
    pay = pd.read_csv(RAW / "payments.csv", dtype={"invoice_id": str})
    primes = pd.read_csv(RAW / "primes.csv", dtype=str)
    truth = []

    # ---------------- G702 as submitted ----------------
    orig = pol[pol.line_source == "CONTRACT"].groupby("po_id").scheduled_value.sum()
    paid = pd.to_numeric(pay.set_index("invoice_id").amount_paid, errors="coerce").fillna(0)
    rows = []
    inv = inv.sort_values(["po_id", "period", "invoice_id"])
    dup_numbers = inv.duplicated(["po_id", "invoice_number"], keep="first")
    inv = inv[~dup_numbers]
    for po_id, g in inv.groupby("po_id"):
        prev_cert = 0.0
        for k, r in enumerate(g.itertuples()):
            approved = co[(co.po_id == po_id) & (co.status == "APPROVED") & (co.approval_period <= r.period)].co_amount.sum()
            pending = co[(co.po_id == po_id) & (co.status == "PENDING")].co_amount.sum()
            claimed_cos, claimed_prev = approved, prev_cert
            rows.append({"invoice_id": r.invoice_id, "po_id": po_id, "application_no": k + 1, "period": r.period,
                         "original_contract_sum": round(orig.get(po_id, 0), 2), "net_change_by_cos": round(claimed_cos, 2),
                         "less_previous_certificates": round(claimed_prev, 2), "approved_cos_on_file": round(approved, 2),
                         "pending_cos_on_file": round(pending, 2), "certified_previous_ledger": round(prev_cert, 2)})
            prev_cert += float(paid.get(r.invoice_id, 0) or 0)
    g702 = pd.DataFrame(rows)
    # plant: previous certificates understated (inflates the current payment due), and pending COs in the contract sum
    late = g702[g702.application_no >= 3]
    for i in rng.choice(late.index, 4, replace=False):
        short = round(float(rng.choice([0.04, 0.06, 0.08])) * g702.at[i, "certified_previous_ledger"], 2)
        g702.at[i, "less_previous_certificates"] = round(g702.at[i, "certified_previous_ledger"] - short, 2)
        truth.append({"invoice_id": g702.at[i, "invoice_id"], "check": "PREVIOUS_CERT_MISMATCH", "amount": short})
    cand = g702[(g702.pending_cos_on_file > 0) & ~g702.index.isin([t for t in g702.index if g702.at[t, "invoice_id"] in
                                                                     {x["invoice_id"] for x in truth}])]
    for i in rng.choice(cand.index, min(3, len(cand)), replace=False):
        g702.at[i, "net_change_by_cos"] = round(g702.at[i, "net_change_by_cos"] + g702.at[i, "pending_cos_on_file"], 2)
        truth.append({"invoice_id": g702.at[i, "invoice_id"], "check": "PENDING_CO_IN_CONTRACT_SUM",
                      "amount": g702.at[i, "pending_cos_on_file"]})
    g702.drop(columns=["certified_previous_ledger"]).to_csv(RAW / "g702_submitted.csv", index=False)

    # ---------------- stored materials ----------------
    sm = []
    mat = {"PRM-E": [("Switchgear lineup, 15kV", 380000), ("MV cable reels", 145000), ("Busway sections", 96000)],
           "PRM-M": [("CRAH units", 410000), ("Chilled-water pipe and fittings", 128000), ("Pumps and VFDs", 88000)],
           "PRM-C": [("Rebar, fabricated", 160000), ("Precast trench sections", 74000), ("Embed plates", 39000)]}
    pick = inv.sample(24, random_state=7)
    for r in pick.itertuples():
        name, val = mat[r.prime_id][int(rng.integers(0, 3))]
        off = bool(rng.random() < 0.5)
        sm.append({"invoice_id": r.invoice_id, "po_id": r.po_id, "description": name, "location": "Off-site bonded warehouse"
                   if off else "On site, secured laydown", "value": round(val * float(rng.uniform(0.6, 1.2)), 2),
                   "bill_of_sale": "Y", "insurance_certificate": "Y", "owner_inspection": "Y", "delivery_receipt": "Y"})
    sm = pd.DataFrame(sm)
    for i in rng.choice(sm.index, 5, replace=False):
        if sm.at[i, "location"].startswith("Off"):
            field = str(rng.choice(["bill_of_sale", "insurance_certificate", "owner_inspection"]))
        else:
            field = "delivery_receipt"
        sm.at[i, field] = "N"
        truth.append({"invoice_id": sm.at[i, "invoice_id"], "check": "STORED_MATERIALS_UNSUPPORTED", "amount": sm.at[i, "value"]})
    sm.to_csv(RAW / "stored_materials.csv", index=False)

    # ---------------- lien waivers ----------------
    subs = {"PRM-E": [("Caprock Low-Voltage (fictional)", 0.18), ("Pecos Cable Pulling (fictional)", 0.12)],
            "PRM-M": [("Sandhill Insulation (fictional)", 0.10), ("Mesa Controls (fictional)", 0.15)],
            "PRM-C": [("Llano Rebar (fictional)", 0.20), ("Draw Concrete Pumping (fictional)", 0.08)]}
    lw = []
    for r in inv.itertuples():
        lw.append({"invoice_id": r.invoice_id, "party": "Prime contractor", "waiver_type": "Conditional, this payment",
                   "amount_covered": round(r.net_amount, 2), "received": "Y"})
        if r.period > "2026-01":
            lw.append({"invoice_id": r.invoice_id, "party": "Prime contractor", "waiver_type": "Unconditional, previous payment",
                       "amount_covered": 0.0, "received": "Y"})
        for name, share in subs[r.prime_id]:
            lw.append({"invoice_id": r.invoice_id, "party": name, "waiver_type": "Conditional, this payment",
                       "amount_covered": round(r.gross_amount * share, 2), "received": "Y"})
    lw = pd.DataFrame(lw)
    sub_rows = lw[(lw.party != "Prime contractor") & (lw.invoice_id.isin(inv[inv.period >= "2026-05"].invoice_id))]
    for i in rng.choice(sub_rows.index, 6, replace=False):
        lw.at[i, "received"] = "N"
        truth.append({"invoice_id": lw.at[i, "invoice_id"], "check": "LIEN_WAIVER_MISSING", "amount": lw.at[i, "amount_covered"]})
    lw.to_csv(RAW / "lien_waivers.csv", index=False)

    # ---------------- certificates of insurance ----------------
    coi = []
    for p in primes.prime_id:
        for cov, lim in [("General liability", 2_000_000), ("Umbrella / excess", 10_000_000), ("Workers' compensation", 1_000_000),
                         ("Auto liability", 1_000_000)]:
            coi.append({"prime_id": p, "coverage": cov, "limit_required": lim, "limit_on_certificate": lim,
                        "expires": "2027-03-31"})
    coi = pd.DataFrame(coi)
    coi.loc[(coi.prime_id == "PRM-M") & (coi.coverage == "Umbrella / excess"), "expires"] = "2026-07-15"
    coi.loc[(coi.prime_id == "PRM-C") & (coi.coverage == "Auto liability"), "limit_on_certificate"] = 500_000
    coi.to_csv(RAW / "insurance_certs.csv", index=False)
    lastc = inv[inv.prime_id == "PRM-C"].sort_values("period").iloc[-1]
    truth.append({"invoice_id": lastc.invoice_id, "check": "INSURANCE_LIMIT_SHORT", "amount": 0.0})
    for r in inv[(inv.prime_id == "PRM-M") & (inv.invoice_date > "2026-07-15")].itertuples():
        truth.append({"invoice_id": r.invoice_id, "check": "INSURANCE_LAPSED", "amount": 0.0})

    # ---------------- cost-plus / GMP package ----------------
    terms = pd.DataFrame([
        {"package_id": f"GMP-{p}", "prime_id": p, "fee_pct": 5.0, "labor_burden_cap_pct": 38.0,
         "gc_budget": b, "contract_months": 12, "start_period": "2026-01", "contingency_budget": cb}
        for p, b, cb in [("PRM-E", 2_400_000, 600_000), ("PRM-M", 2_100_000, 500_000), ("PRM-C", 2_900_000, 700_000)]])
    terms.to_csv(RAW / "gmp_terms.csv", index=False)
    costs, n = [], 0
    periods = sorted(inv.period.unique())
    for t in terms.itertuples():
        for m, per in enumerate(periods):
            n += 1
            gc = round(t.gc_budget / t.contract_months * float(rng.uniform(0.92, 1.05)), 2)
            base_l = round(float(rng.uniform(90_000, 140_000)), 2)
            sub_cost = round(float(rng.uniform(150_000, 260_000)), 2)
            items = [
                ("General conditions", "Site office, supervision, temporary utilities", gc, None, "", ""),
                ("Labor", "Direct craft labor (reimbursable)", base_l, 36.0, "", ""),
                ("Subcontract", "Subcontract costs (sub fee excluded)", sub_cost, None, "N", ""),
            ]
            for cat, desc, cost, burden, sub_fee_incl, appr in items:
                burden_amt = round(cost * burden / 100, 2) if burden else 0.0
                costs.append({"cost_id": f"GC-{n:03d}-{len(costs) % 9}", "package_id": t.package_id, "prime_id": t.prime_id,
                              "period": per, "category": cat, "description": desc, "base_cost": cost,
                              "labor_burden_pct": burden if burden else np.nan, "burden_amount": burden_amt,
                              "sub_fee_included": sub_fee_incl, "sub_fee_amount": 0.0, "approval_ref": appr,
                              "budget_value": np.nan, "award_value": np.nan,
                              "fee_amount": round((cost + burden_amt) * t.fee_pct / 100, 2)})
    gm = pd.DataFrame(costs)
    gm["cost_id"] = [f"CP-{i + 1:03d}" for i in range(len(gm))]

    def plant(idx, check, amount):
        truth.append({"invoice_id": gm.at[idx, "cost_id"], "check": check, "amount": round(amount, 2)})

    lab = gm[gm.category == "Labor"].index
    for i in rng.choice(lab, 3, replace=False):
        gm.at[i, "labor_burden_pct"] = 46.0
        gm.at[i, "burden_amount"] = round(gm.at[i, "base_cost"] * 0.46, 2)
        gm.at[i, "fee_amount"] = round((gm.at[i, "base_cost"] + gm.at[i, "burden_amount"]) * 0.05, 2)
        over = gm.at[i, "base_cost"] * (0.46 - 0.38)
        plant(i, "BURDEN_OVER_CAP", over * 1.05)
    sub = gm[gm.category == "Subcontract"].index
    for i in rng.choice(sub, 3, replace=False):
        sf = round(gm.at[i, "base_cost"] * 0.10, 2)
        gm.at[i, "sub_fee_included"] = "Y"
        gm.at[i, "sub_fee_amount"] = sf
        gm.at[i, "description"] = "Subcontract costs (includes sub's 10% fee)"
        plant(i, "MARKUP_ON_MARKUP", sf * 0.05)
    extra = []
    for p, per, cat, desc, cost, appr in [
        ("PRM-E", "2026-06", "Bonus", "Project team completion bonus", 42_000, ""),
        ("PRM-M", "2026-07", "Legal", "Outside counsel, claim preparation", 28_500, ""),
        ("PRM-C", "2026-05", "Rework", "Re-pour of rejected slab section (contractor error)", 61_300, ""),
        ("PRM-E", "2026-04", "Contingency draw", "Draw for conduit reroute", 88_000, "OWN-CA-004"),
        ("PRM-M", "2026-08", "Contingency draw", "Draw for added pipe supports", 54_000, ""),
        ("PRM-C", "2026-03", "Subcontract buyout", "Site utilities subcontract, billed at budget", 0, ""),
    ]:
        r = {"cost_id": "", "package_id": f"GMP-{p}", "prime_id": p, "period": per, "category": cat, "description": desc,
             "base_cost": cost, "labor_burden_pct": np.nan, "burden_amount": 0.0, "sub_fee_included": "", "sub_fee_amount": 0.0,
             "approval_ref": appr, "budget_value": np.nan, "award_value": np.nan, "fee_amount": round(cost * 0.05, 2)}
        if cat == "Subcontract buyout":
            r.update(budget_value=1_250_000, award_value=1_118_000, base_cost=1_250_000, fee_amount=62_500.0)
        extra.append(r)
    gm = pd.concat([gm, pd.DataFrame(extra)], ignore_index=True)
    gm["cost_id"] = [f"CP-{i + 1:03d}" for i in range(len(gm))]
    for i in gm.index[-6:]:
        r = gm.loc[i]
        if r.category in ("Bonus", "Legal", "Rework"):
            plant(i, "UNALLOWABLE_COST", r.base_cost + r.fee_amount)
        elif r.category == "Contingency draw" and not r.approval_ref:
            plant(i, "CONTINGENCY_NOT_APPROVED", r.base_cost + r.fee_amount)
        elif r.category == "Subcontract buyout":
            plant(i, "BUYOUT_SAVINGS_KEPT", (r.budget_value - r.award_value) * 1.05)
    # general conditions billed ahead of time for one prime: front-load the first months
    fl = gm[(gm.prime_id == "PRM-C") & (gm.category == "General conditions") & (gm.period <= "2026-04")].index
    gm.loc[fl, "base_cost"] = (gm.loc[fl, "base_cost"] * 1.6).round(2)
    gm.loc[fl, "fee_amount"] = (gm.loc[fl, "base_cost"] * 0.05).round(2)
    truth.append({"invoice_id": "GMP-PRM-C", "check": "GC_AHEAD_OF_TIME", "amount": np.nan})
    gm["billed_amount"] = (gm.base_cost + gm.burden_amount + gm.fee_amount).round(2)
    gm.to_csv(RAW / "gmp_costs.csv", index=False)

    pd.DataFrame(truth).to_csv(RAW / "payapp_truth.csv", index=False)
    print(f"g702 {len(g702)} · stored {len(sm)} · waivers {len(lw)} · insurance {len(coi)} · gmp costs {len(gm)} · "
          f"planted {len(truth)}")


if __name__ == "__main__":
    main()
