"""
Turn engine results into management information:
  * project_summary()  - one row per project: contract value, billed, verified, $ at risk, status
  * prime_trend()      - first-pass rate by contractor by month
  * rate_spread()      - same item, same contractor, different price across projects
  * generate_insights()- plain-English findings, each with a $ amount and a recommended action

Everything works on whatever subset of projects / contractors / months is passed in.
"""
import numpy as np
import pandas as pd

RULE_LABEL = {
    "QTY_OVER_RECEIPT": "quantities billed above field-verified",
    "RATE_OVER_CONTRACT": "unit rates above contract",
    "PCT_COMPLETE_OVERBILL": "lump-sum progress billed ahead of verified progress",
    "UNAPPROVED_CO": "work billed on change orders that aren't approved",
    "TM_OVER_APPROVED": "T&M hours above the pre-approved hours",
    "TM_NOT_PREAPPROVED": "T&M hours with no pre-approval",
    "PER_DIEM_OVER_POLICY": "per diem above the policy cap",
    "RETAINAGE_ERROR": "retainage withheld incorrectly",
    "EXTENSION_ERROR": "arithmetic errors (quantity x rate does not equal amount)",
    "DUPLICATE_INVOICE": "duplicate invoice submissions",
    "NO_RECEIPT": "work billed with no field verification",
    "HEADER_MISMATCH": "invoice totals that don't add up",
}
RULE_ACTION = {
    "QTY_OVER_RECEIPT": "Ask the field team to re-measure the flagged items and pay verified quantities only.",
    "RATE_OVER_CONTRACT": "Pay at the contract rate and reject the difference.",
    "PCT_COMPLETE_OVERBILL": "Pay the verified % complete only, and agree how progress is measured for lump-sum items.",
    "UNAPPROVED_CO": "Remove these lines until the change orders are approved, and remind the contractor of the pre-approval rule.",
    "TM_OVER_APPROVED": "Pay approved hours only; extra hours need approval before the work, not after.",
    "TM_NOT_PREAPPROVED": "Hold until a pre-approval is on file.",
    "PER_DIEM_OVER_POLICY": "Pay at the policy cap and restate the per diem policy to the contractor.",
    "RETAINAGE_ERROR": "Correct the retainage before payment.",
    "EXTENSION_ERROR": "Return the invoice for correction and ask for system-generated pay applications.",
    "DUPLICATE_INVOICE": "Reject the duplicates and require one pay-application number per contract per month.",
    "NO_RECEIPT": "Hold until the field team verifies the work.",
    "HEADER_MISMATCH": "Return the invoice; totals must reconcile to the lines.",
}


def money(x):
    x = float(x or 0)
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def prime_names(tables):
    p = tables.get("primes")
    if p is None or not len(p):
        return {}
    return dict(zip(p["prime_id"], p["prime_name"].str.replace(" (fictional)", "", regex=False)))


def filter_scope(tables, gates, exceptions, projects=None, primes=None, periods=None):
    """Apply project / contractor / month filters. Returns filtered gates, exceptions, invoice ids."""
    g = gates.copy()
    if projects:
        g = g[g.project_id.isin(projects)]
    if primes:
        g = g[g.prime_id.isin(primes)]
    if periods:
        g = g[g.period.isin(periods)]
    ids = set(g.invoice_id)
    return g, exceptions[exceptions.invoice_id.isin(ids)], ids


def project_summary(tables, gates, invoice_ids, projects=None, primes=None, periods=None):
    po = tables["purchase_orders"][["po_id", "project_id", "prime_id"]]
    if projects:
        po = po[po.project_id.isin(projects)]
    if primes:
        po = po[po.prime_id.isin(primes)]
    pol = tables["po_lines"].merge(po, on="po_id")
    contract = pol.groupby("project_id").scheduled_value.sum().rename("contract_value")
    sov_contract = pol[pol.line_source == "CONTRACT"].groupby("project_id").scheduled_value.sum().rename("sov_contract")

    # Billed, excluding invoices flagged as duplicates (they'd double count)
    dup_ids = set(gates.loc[gates.rules_triggered.fillna("").str.contains("DUPLICATE_INVOICE"), "invoice_id"])
    inv = tables["invoices"][tables["invoices"].invoice_id.isin(invoice_ids - dup_ids)][["invoice_id", "po_id"]]
    lines = tables["invoice_lines"].merge(inv, on="invoice_id").merge(po, on="po_id")
    work = lines[lines.line_type.isin(["SOV", "CO"])].groupby("project_id").amount.sum().rename("billed_work")
    sov_billed = lines[lines.line_type == "SOV"].groupby("project_id").amount.sum().rename("billed_sov")
    other = lines[lines.line_type.isin(["TM", "PER_DIEM"])].groupby("project_id").amount.sum().rename("billed_tm_per_diem")

    # Field-verified value of SOV work in place
    gr = tables["goods_receipts"].merge(po, on="po_id")
    if periods:
        gr = gr[gr.period.isin(periods)]
    gr = gr.merge(tables["po_lines"][["po_id", "item_code", "contract_unit_rate", "scheduled_value"]],
                  on=["po_id", "item_code"], how="left")
    unit = gr[gr.verified_qty.notna()]
    v_unit = (unit.verified_qty * unit.contract_unit_rate).groupby(unit.project_id).sum()
    ls = gr[gr.verified_pct_complete_cum.notna()].sort_values("period")
    ls_last = ls.groupby(["project_id", "po_id", "item_code"]).agg(
        pct=("verified_pct_complete_cum", "last"), sv=("scheduled_value", "first")).reset_index()
    v_ls = (ls_last.pct * ls_last.sv / 100).groupby(ls_last.project_id).sum()
    verified = v_unit.add(v_ls, fill_value=0).rename("verified_sov")

    co = tables["change_orders"].merge(po[["po_id", "project_id"]], on="po_id", suffixes=("", "_po"))
    pending = co[co.status == "PENDING"].groupby("project_id").co_amount.sum().rename("pending_co")

    g = gates[gates.invoice_id.isin(invoice_ids)]
    gs = g.groupby("project_id").agg(
        invoices=("invoice_id", "count"),
        billed_gross=("gross_amount", "sum"),
        amount_at_risk=("amount_at_risk", "sum"),
        first_pass_pct=("gate", lambda s: round(100 * (s == "PASS").mean(), 0)),
        on_hold=("gate", lambda s: int((s == "HOLD").sum())),
        last_period=("period", "max"),
    )
    df = pd.concat([contract, sov_contract, work, sov_billed, other, verified, pending, gs], axis=1).fillna(0)
    df = df[df.invoices > 0]
    df["pct_billed"] = (100 * df.billed_work / df.contract_value.where(df.contract_value > 0)).round(1)
    df["pct_verified"] = (100 * df.verified_sov / df.sov_contract.where(df.sov_contract > 0)).round(1)
    df["billed_ahead_of_verified"] = (df.billed_sov - df.verified_sov).round(0)
    df["risk_pct"] = (100 * df.amount_at_risk / df.billed_gross.where(df.billed_gross > 0)).round(1)
    gap_pct = 100 * df.billed_ahead_of_verified / df.sov_contract.where(df.sov_contract > 0)
    df["status"] = np.where((df.risk_pct > 5) | (gap_pct > 3), "Red",
                            np.where((df.risk_pct > 2) | (gap_pct > 1), "Amber", "Green"))
    campus = tables.get("projects")
    if campus is not None and "campus" in campus.columns:
        df = df.join(campus.set_index("project_id")["campus"])
    return df.reset_index().rename(columns={"index": "project_id"})


def prime_trend(gates):
    return (gates.groupby(["period", "prime_id"]).gate
            .apply(lambda s: round(100 * (s == "PASS").mean(), 0)).rename("first_pass_pct").reset_index())


def rate_spread(tables, projects=None, primes=None):
    po = tables["purchase_orders"][["po_id", "project_id", "prime_id"]]
    if projects:
        po = po[po.project_id.isin(projects)]
    if primes:
        po = po[po.prime_id.isin(primes)]
    pol = tables["po_lines"].merge(po, on="po_id")
    pol = pol[(pol.line_source == "CONTRACT") & (pol.unit.str.upper() != "LS")]
    rows = []
    for (prm, item), grp in pol.groupby(["prime_id", "item_code"]):
        if grp.project_id.nunique() < 2:
            continue
        lo, hi = grp.contract_unit_rate.min(), grp.contract_unit_rate.max()
        excess = ((grp.contract_unit_rate - lo) * grp.contract_qty).sum()
        rows.append({"prime_id": prm, "item_code": item, "description": grp.description.iloc[0],
                     "unit": grp.unit.iloc[0], "lowest_rate": lo, "highest_rate": hi,
                     "lowest_at": grp.loc[grp.contract_unit_rate.idxmin(), "project_id"],
                     "highest_at": grp.loc[grp.contract_unit_rate.idxmax(), "project_id"],
                     "spread_pct": round(100 * (hi / lo - 1), 1), "excess_vs_lowest": round(excess, 0)})
    return pd.DataFrame(rows).sort_values("excess_vs_lowest", ascending=False) if rows else pd.DataFrame()


def generate_insights(tables, gates_f, exc_f, summary, trend, spread, top_n=7):
    names = prime_names(tables)
    nm = lambda p: names.get(p, p)
    poss = lambda p: nm(p) + ("'" if nm(p).endswith("s") else "'s")
    out = []

    # 1. Where the money at risk is concentrated (contractor x issue)
    e = exc_f[exc_f.amount_at_risk > 0].merge(gates_f[["invoice_id", "prime_id", "project_id"]], on="invoice_id")
    total_risk = gates_f.amount_at_risk.sum()
    if len(e) and total_risk > 0:
        for rule, grp in sorted(e.groupby("rule"), key=lambda kv: -kv[1].amount_at_risk.sum())[:3]:
            amt = grp.amount_at_risk.sum()
            by_prime = grp.groupby("prime_id").amount_at_risk.sum().sort_values(ascending=False)
            top_p = by_prime.index[0]
            who = (f"mostly {nm(top_p)} ({money(by_prime.iloc[0])})" if len(by_prime) > 1 else nm(top_p))
            out.append({"level": "critical", "amount": amt,
                        "title": f"Hold {money(amt)}: {RULE_LABEL.get(rule, rule)}",
                        "detail": f"{100 * amt / total_risk:.0f}% of all money at risk in this view. "
                                  f"{grp.invoice_id.nunique()} invoice(s) across {grp.project_id.nunique()} building(s); "
                                  f"{who}; most at {grp.project_id.value_counts().index[0]}.",
                        "action": RULE_ACTION.get(rule, "Review before payment.")})

    # 2. Project hotspot
    if len(summary) > 1 and summary.billed_gross.sum() > 0:
        avg = 100 * summary.amount_at_risk.sum() / summary.billed_gross.sum()
        h = summary.sort_values("risk_pct", ascending=False).iloc[0]
        if h.risk_pct > 1.5 * avg and h.amount_at_risk > 0:
            out.append({"level": "warning", "amount": h.amount_at_risk,
                        "title": f"Focus the next audit on {h.project_id}",
                        "detail": f"{h.risk_pct:.1f}% of its billing is at risk ({money(h.amount_at_risk)}), "
                                  f"versus {avg:.1f}% across the projects in view.",
                        "action": f"Walk the site with the field team and review every held invoice at {h.project_id} first."})

    # 3. Cumulative billing ahead of verified work
    if len(summary):
        s = summary.copy()
        s["gap_pct"] = 100 * s.billed_ahead_of_verified / s.sov_contract.where(s.sov_contract > 0)
        s = s[s.gap_pct > 1].sort_values("billed_ahead_of_verified", ascending=False)
        if len(s):
            r = s.iloc[0]
            out.append({"level": "warning", "amount": r.billed_ahead_of_verified,
                        "title": f"{r.project_id} has billed {money(r.billed_ahead_of_verified)} more than the work verified in place",
                        "detail": f"Billed {money(r.billed_sov)} of contract work vs {money(r.verified_sov)} verified "
                                  f"({r.gap_pct:.1f}% of the contract value).",
                        "action": "Reconcile cumulative quantities with the field team before the next pay application; "
                                  "withhold the difference if it persists."})

    # 4. Change-order exposure
    if len(summary) and summary.pending_co.sum() > 0:
        co = tables["change_orders"]
        po = tables["purchase_orders"][["po_id", "project_id", "prime_id"]]
        co = co.merge(po, on="po_id", suffixes=("", "_po"))
        co = co[(co.status == "PENDING") & co.project_id.isin(summary.project_id)]
        if len(co):
            top = co.groupby("prime_id").co_amount.sum().sort_values(ascending=False)
            out.append({"level": "info", "amount": co.co_amount.sum(),
                        "title": f"{money(co.co_amount.sum())} of change orders are waiting for a decision",
                        "detail": f"{len(co)} pending change order(s); largest share from {nm(top.index[0])} ({money(top.iloc[0])}).",
                        "action": "Approve or reject them before the work is billed, and add the likely approvals "
                                  "to the cost forecast now."})

    # 5. Trend: contractor getting worse, or a bad month for everyone
    if len(trend) and trend.period.nunique() >= 4:
        per = sorted(trend.period.unique())
        early, late = per[:-2], per[-2:]
        worst = None
        for prm, grp in trend.groupby("prime_id"):
            a = grp[grp.period.isin(early)].first_pass_pct.mean()
            b = grp[grp.period.isin(late)].first_pass_pct.mean()
            if a - b >= 15 and (worst is None or a - b > worst[1] - worst[2]):
                worst = (prm, a, b)
        if worst:
            out.append({"level": "warning", "amount": 0,
                        "title": f"{poss(worst[0])} billing quality is getting worse",
                        "detail": f"First-pass rate fell from {worst[1]:.0f}% (earlier months) to {worst[2]:.0f}% (last two months).",
                        "action": "Hold a billing-quality meeting with the contractor's project controls lead "
                                  "and review how they prepare pay applications."})
        m = trend.groupby("period").agg(fp=("first_pass_pct", "mean"), k=("prime_id", "nunique"))
        bad = m[(m.fp < 25) & (m.k >= 2)].fp
        if len(bad):
            out.append({"level": "warning", "amount": 0,
                        "title": f"Systemic problem in {', '.join(bad.index[:2])}",
                        "detail": f"Average first-pass rate across contractors was only {bad.iloc[0]:.0f}%.",
                        "action": "Look for a shared cause (a new rule, a template change, a data feed issue) "
                                  "rather than chasing each contractor."})

    # 6. Same item, different price
    if len(spread):
        r = spread.iloc[0]
        if r.spread_pct >= 5:
            out.append({"level": "info", "amount": r.excess_vs_lowest,
                        "title": f"{nm(r.prime_id)} charges {r.spread_pct:.0f}% more for {r.description.lower()} at {r.highest_at} than at {r.lowest_at}",
                        "detail": f"${r.lowest_rate:,.2f} vs ${r.highest_rate:,.2f} per {r.unit}. "
                                  f"Pricing every building at the lowest rate would be {money(r.excess_vs_lowest)} less.",
                        "action": "Use the lowest rate as the benchmark when pricing change orders and the next contract."})

    # 7. Best performer (good news drives decisions too)
    if gates_f.prime_id.nunique() > 1:
        fp = gates_f.groupby("prime_id").agg(n=("gate", "size"), fp=("gate", lambda s: 100 * (s == "PASS").mean()))
        fp = fp[fp.n >= 5].sort_values("fp", ascending=False)
        if len(fp) and fp.fp.iloc[0] >= 60:
            out.append({"level": "good", "amount": 0,
                        "title": f"{nm(fp.index[0])} is the most reliable biller ({fp.fp.iloc[0]:.0f}% pass first time)",
                        "detail": f"Based on {int(fp.n.iloc[0])} invoices in this view.",
                        "action": "Consider a lighter review for their low-risk lines, and share their pay-app "
                                  "format with the other contractors."})

    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    out.sort(key=lambda d: (order[d["level"]], -d["amount"]))
    return out[:top_n]
