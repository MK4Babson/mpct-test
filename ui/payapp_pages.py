"""Pay applications (G702/G703, compliance, cost-plus/GMP, withholding notice) and the Issues & decisions workspace."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

import decisions as D
import payapp as PA
from insights import RULE_ACTION, RULE_LABEL, money
from ui.cost_pages import hero, howto

LABELS = {**RULE_LABEL, **PA.LABEL}
STATUS_ICON = {"Detected": "○", "Assigned": "◔", "Investigating": "◑", "Decided": "◕", "Reconciled": "●", "Closed": "✓"}
ACTIONS = {**RULE_ACTION, **PA.ACTION}


@st.cache_data(show_spinner=False)
def _pa(folder):
    return PA.load(folder) if folder else None


def _pa_ctx(ctx):
    return _pa(str(ctx.data_folder)) if getattr(ctx, "data_folder", None) else None


def findings(ctx):
    """Every finding for the program: bill checks + pay-application + cost-plus checks, one schema."""
    g = ctx.gates[["invoice_id", "prime_id", "period"]]
    ex = ctx.exceptions.merge(g, on="invoice_id", how="left").assign(source="Bill check")
    pa = _pa_ctx(ctx)
    parts = [ex[["invoice_id", "line_no", "rule", "severity", "detail", "amount_at_risk", "prime_id", "period", "source"]]]
    if pa is not None:
        parts += [PA.checks(ctx.tables, pa), PA.gmp_checks(pa).assign(source="Cost-plus (GMP)")]
    f = pd.concat([p for p in parts if len(p)], ignore_index=True)
    names = ctx.tables["primes"].set_index("prime_id").prime_name.str.replace(" (fictional)", "", regex=False) \
        if "primes" in ctx.tables else pd.Series(dtype=str)
    f["contractor"] = f.prime_id.map(names).fillna(f.prime_id)
    f["amount_at_risk"] = f.amount_at_risk.fillna(0).clip(lower=0)
    return f


# ---------------------------------------------------------------- Pay applications
def payapp_page(ctx):
    hero("Pay applications · AIA G702/G703 style", "Is the contractor owed what the pay application asks for?",
         "The owner's own ledger next to the contractor's figures: work to date, stored materials, retainage and what was "
         "already certified. Then the documents that must come with every bill, and the cost-plus checks.")
    pa = _pa_ctx(ctx)
    if pa is None:
        st.info("This program has no US pay-application data. The North Koel canal uses Indian item-rate running bills, "
                "checked on the Program analytics page. Switch to the West Texas program to see this page.")
        return
    t = ctx.tables
    allf = findings(ctx)
    pf = allf[allf.source != "Bill check"]
    k = st.columns(4)
    k[0].metric("Pay-app findings", len(pf[pf.source == "Pay application"]))
    k[1].metric("Cost-plus findings", len(pf[pf.source == "Cost-plus (GMP)"]))
    k[2].metric("Flagged", money(pf.amount_at_risk.sum()), help="Sum of findings. Some overlap with the bill checks.")
    k[3].metric("Document holds", int(((pf.amount_at_risk == 0) & (pf.severity == "HOLD")).sum()),
                help="No dollar amount, but payment should wait for a document (e.g. an expired insurance certificate).")
    tabs = st.tabs(["G702/G703 ledger", "Stored materials", "Lien waivers & insurance", "Cost-plus & GMP", "Withholding notice",
                    "Accuracy"])
    led = PA.ledger(t)
    labels = {r.invoice_id: f"{r.invoice_number} · app #{r.application_no} · {r.period}" for r in led.itertuples()}

    with tabs[0]:
        mism = set(pf[pf.rule.isin(["PREVIOUS_CERT_MISMATCH", "PENDING_CO_IN_CONTRACT_SUM"])].invoice_id)
        opts = sorted(led.invoice_id, key=lambda i: (i not in mism, labels[i]))
        pick = st.selectbox("Pay application", opts, format_func=lambda i: ("⚑ " if i in mism else "") + labels[i],
                            help="⚑ = the contractor's G702 differs from the owner's ledger.")
        df, r = PA.g702(t, pa, pick)
        st.markdown("##### G702 summary: owner's ledger vs contractor's figures")
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "line": "Line", "owner": st.column_config.NumberColumn("Owner's ledger", format="$%,.0f"),
            "contractor": st.column_config.NumberColumn("Contractor submitted", format="$%,.0f"),
            "difference": st.column_config.NumberColumn("Difference", format="$%,.0f")})
        diff = df.set_index("line").difference
        due_gap = diff.filter(like="8.").iloc[0]
        if pd.notna(due_gap) and abs(due_gap) > 1:
            st.error(f"The contractor's current payment due is **{money(due_gap)}** {'higher' if due_gap > 0 else 'lower'} than the "
                     "owner's ledger supports. Line 7 comes from the owner's certified-payment record, never from the contractor.")
        howto("<b>How it works:</b> previous certificates are what the owner actually certified on earlier applications, "
              "from the payment record. If a contractor lists less, the current payment due grows by the same amount. "
              "Retainage is held on work and stored materials at the contract rate.")
        st.markdown("##### G703 continuation sheet")
        sheet = PA.g703(t, pa, pick)
        st.dataframe(sheet, hide_index=True, width="stretch", column_config={
            "item": "Item", "description": "Description",
            "scheduled_value": st.column_config.NumberColumn("Scheduled value", format="$%,.0f"),
            "from_previous": st.column_config.NumberColumn("From previous", format="$%,.0f"),
            "this_period": st.column_config.NumberColumn("This period", format="$%,.0f"),
            "stored": st.column_config.NumberColumn("Stored", format="$%,.0f"),
            "total_to_date": st.column_config.NumberColumn("Total to date", format="$%,.0f"),
            "pct": st.column_config.NumberColumn("% complete", format="%.0f%%"),
            "balance_to_finish": st.column_config.NumberColumn("Balance to finish", format="$%,.0f")})
        mine = pf[pf.invoice_id == pick]
        if len(mine):
            st.markdown("##### Findings on this pay application")
            for x in mine.itertuples():
                st.markdown(f"- **{LABELS.get(x.rule, x.rule)}** ({x.severity.lower()}, {money(x.amount_at_risk)}): {x.detail}")

    with tabs[1]:
        sm = pa["stored_materials"].copy()
        need = lambda r: ["bill_of_sale", "insurance_certificate", "owner_inspection"] if r.location.startswith("Off") else ["delivery_receipt"]  # noqa: E731
        sm["status"] = ["Supported" if all(r[n] == "Y" for n in need(r)) else "Missing evidence" for _, r in sm.iterrows()]
        sm["bill"] = sm.invoice_id.map(labels)
        st.dataframe(sm[["bill", "description", "location", "value", "bill_of_sale", "insurance_certificate", "owner_inspection",
                         "delivery_receipt", "status"]].sort_values("status"), hide_index=True, width="stretch",
                     column_config={"value": st.column_config.NumberColumn("Value", format="$%,.0f")})
        howto("<b>Rule:</b> materials stored off site are paid only with a bill of sale (title to the owner), proof of insurance and "
              "an owner inspection. Materials stored on site need a delivery receipt. Without them, the owner pays for goods it may "
              "not own or that may not exist.")

    with tabs[2]:
        lw = pa["lien_waivers"]
        miss = lw[lw.received != "Y"].assign(bill=lambda d: d.invoice_id.map(labels))
        c1, c2 = st.columns(2)
        c1.metric("Waivers expected", len(lw))
        c2.metric("Missing", len(miss), money(miss.amount_covered.sum()) + " uncovered", delta_color="off", delta_arrow="off")
        st.dataframe(miss[["bill", "party", "waiver_type", "amount_covered"]], hide_index=True, width="stretch",
                     column_config={"amount_covered": st.column_config.NumberColumn("Amount covered", format="$%,.0f")})
        howto("<b>Why it matters:</b> if the owner pays the prime and the prime does not pay a subcontractor, the sub can place a "
              "lien on the owner's property. Without a waiver the owner can end up paying twice. Common fixes: hold the uncovered "
              "amount, or pay by joint check.")
        st.markdown("##### Certificates of insurance")
        coi = pa["insurance_certs"].copy()
        names = t["primes"].set_index("prime_id").prime_name.str.replace(" (fictional)", "", regex=False)
        coi["contractor"] = coi.prime_id.map(names)
        today_ = led.invoice_date.max()
        coi["status"] = np.where(coi.expires < today_, "Expired", np.where(coi.limit_on_certificate < coi.limit_required,
                                                                            "Limit too low", "OK"))
        st.dataframe(coi[["contractor", "coverage", "limit_required", "limit_on_certificate", "expires", "status"]],
                     hide_index=True, width="stretch", column_config={
                         "limit_required": st.column_config.NumberColumn("Required", format="$%,.0f"),
                         "limit_on_certificate": st.column_config.NumberColumn("On certificate", format="$%,.0f")})

    with tabs[3]:
        gm, terms = pa["gmp_costs"], pa["gmp_terms"]
        gf = pf[pf.source == "Cost-plus (GMP)"]
        st.markdown("Each prime's **general conditions and reimbursables** are billed cost-plus, with a fee and a guaranteed "
                    "maximum. These checks are where cost-plus overbilling usually hides.")
        s = gf.groupby("rule").amount_at_risk.agg(["count", "sum"]).reset_index()
        s["check"] = s.rule.map(LABELS)
        s["next step"] = s.rule.map(ACTIONS)
        st.dataframe(s[["check", "count", "sum", "next step"]], hide_index=True, width="stretch",
                     column_config={"sum": st.column_config.NumberColumn("Flagged", format="$%,.0f"), "count": "Findings"})
        st.dataframe(gf[["invoice_id", "contractor", "period", "rule", "detail", "amount_at_risk"]].assign(
            rule=lambda d: d.rule.map(LABELS)), hide_index=True, width="stretch", column_config={
            "invoice_id": "Cost line", "rule": "Check", "amount_at_risk": st.column_config.NumberColumn("Flagged", format="$%,.0f")})
        with st.expander("Contract terms used"):
            st.dataframe(terms, hide_index=True, width="stretch")
        howto("<b>The six checks:</b> labor burden above the agreed rate · the prime's fee charged on top of a subcontractor's fee "
              "· costs the contract does not reimburse (bonuses, legal fees, the contractor's own rework) · contingency spent "
              "without owner approval · buyout savings kept instead of returned · general conditions billed faster than time passes.")

    with tabs[4]:
        held = allf[(allf.severity == "HOLD") & (allf.amount_at_risk > 0) & allf.invoice_id.isin(led.invoice_id)]
        bills = held.groupby("invoice_id").amount_at_risk.sum().sort_values(ascending=False)
        if bills.empty:
            st.success("No bill has a recommended hold.")
        else:
            pick = st.selectbox("Bill", list(bills.index), format_func=lambda i: f"{labels.get(i, i)} · {money(bills[i])} recommended hold")
            items = held[held.invoice_id == pick]
            bill = t["invoices"].set_index("invoice_id").loc[pick].to_dict() | {"invoice_id": pick}
            bill["net_amount"] = float(bill["net_amount"])
            st.dataframe(items.assign(reason=items.rule.map(LABELS))[["line_no", "reason", "detail", "amount_at_risk"]],
                         hide_index=True, width="stretch", column_config={"amount_at_risk": st.column_config.NumberColumn(
                             "Amount", format="$%,.0f")})
            st.markdown(f"Requested **{money(bill['net_amount'])}** · withhold **{money(items.amount_at_risk.sum())}** · "
                        f"pay the undisputed **{money(max(bill['net_amount'] - items.amount_at_risk.sum(), 0))}**")
            c1, c2 = st.columns(2)
            owner = c1.text_input("Owner name on the notice", "[Owner]")
            signer = c2.text_input("Signed by", "[Name, title]")
            contractor = allf[allf.invoice_id == pick].contractor.iloc[0]
            doc = PA.notice_docx(bill, [{"rule": x.rule, "detail": x.detail, "amount": x.amount_at_risk,
                                         "ref": f"Line {int(x.line_no)}" if pd.notna(x.line_no) and str(x.line_no).replace('.', '', 1).isdigit()
                                         else "G702"} for x in items.itertuples()], owner_name=owner, contractor=contractor, signer=signer)
            st.download_button("Download the draft notice (Word)", doc, f"withholding_notice_{bill['invoice_number']}.docx",
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            st.caption("A draft for the contract administrator and counsel. Most US states require written notice with reasons "
                       "within a set number of days when payment is withheld; check the contract and the state's prompt-payment law.")

    with tabs[5]:
        acc = PA.accuracy(pf, pa)
        st.dataframe(acc, hide_index=True, width="stretch")
        st.caption("Planted errors with an answer key, as for the bill checks: a perfect catch rate proves the code runs, not that it "
                   "works on real pay applications. 'Found, not planted' are real patterns in the base data (lump sums billed past "
                   "100%, quantity overruns), not false alarms: a reviewer decides each one.")


# ---------------------------------------------------------------- Issues & decisions
def issues_page(ctx):
    hero("Issues & decisions · the decision record", "From flag to decision to payment",
         "Every finding becomes an issue with an owner and a due date. A person investigates, decides and records why. "
         "Every step is kept, so any past decision can be read back.")
    prog = ctx.source_key
    f = findings(ctx)
    added = D.sync(prog, f)
    if added:
        st.toast(f"{added} new issues detected")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.4])
    hours = c1.number_input("Reviewer hours spent", 0.0, 10000.0, 0.0, 10.0, help="To net review cost out of money saved.")
    rate = c2.number_input("Cost per hour ($)", 0.0, 500.0, 90.0, 5.0)
    if c3.button("Load demo decisions", help="Fills realistic reviewer decisions, labelled 'demo', so the measures can be seen."):
        D.load_demo(prog)
        st.rerun()
    if c4.button("Clear all decisions", help="Deletes this program's issues and history, then re-detects from the checks."):
        D.reset(prog)
        st.rerun()
    m = D.measures(prog, hours, rate)
    k = st.columns(5)
    k[0].metric("1. Flagged", money(m["flagged"]), f"{m['total']} issues", delta_color="off", delta_arrow="off",
                help="Sum of all findings. Findings from different checks can overlap on the same dollars.")
    k[1].metric("2. Under review", money(m["under_review"]), help="Flagged amounts not yet decided, or disputed / escalated.")
    k[2].metric("3. Confirmed adjustment", money(m["confirmed"]), f"{m['decided']} decided", delta_color="off", delta_arrow="off")
    k[3].metric("4. Money saved", money(m["saved"]), help="Confirmed adjustments already applied to a payment.")
    k[4].metric("Net of review cost", money(m["net"]), f"{m['overrides']} overrides", delta_color="off", delta_arrow="off")
    df = D.issues(prog)
    if df.empty:
        st.info("No issues.")
        return
    demo = (D.events(prog).user.str.startswith("demo:")).any()
    if demo:
        st.caption("Includes demo decisions (users marked 'demo:'). Clear them before using real decisions.")

    fc = st.columns(4)
    status = fc[0].multiselect("Status", D.STATUSES, default=["Detected", "Assigned", "Investigating"])
    source = fc[1].multiselect("Source", sorted(df.source.unique()), default=sorted(df.source.unique()))
    who = fc[2].multiselect("Contractor", sorted(df.contractor.dropna().unique()), default=sorted(df.contractor.dropna().unique()))
    owner = fc[3].multiselect("Owner", sorted([o for o in df.owner.unique() if o]))
    q = df[df.status.isin(status) & df.source.isin(source) & df.contractor.isin(who)]
    if owner:
        q = q[q.owner.isin(owner)]
    q = q.sort_values(["flagged"], ascending=False)
    view = q.assign(check=q.rule.map(LABELS).fillna(q.rule).str.capitalize(),
                    recommended=q.recommended.map({"HOLD": "⛔ Hold", "REVIEW": "⚠️ Review"}).fillna(q.recommended),
                    status=q.status.map(STATUS_ICON).fillna("") + " " + q.status)
    st.dataframe(view[["issue_id", "invoice_id", "contractor", "check", "flagged", "recommended", "status", "owner", "due", "decision"]],
                 hide_index=True, width="stretch", height=300, column_config={
                     "issue_id": "Issue", "invoice_id": "Bill / line", "contractor": "Contractor", "check": "What is wrong",
                     "flagged": st.column_config.NumberColumn("Flagged", format="$%,.0f"), "recommended": "Recommended",
                     "status": "Status", "owner": "Owner", "due": "Due", "decision": "Decision"})
    if q.empty:
        return
    pick = st.selectbox("Open an issue", list(q.issue_id), format_func=lambda i: (
        lambda r: f"{i} · {r.invoice_id} · {LABELS.get(r.rule, r.rule)} · {money(r.flagged)} · {r.status}")(q.set_index("issue_id").loc[i]))
    r = df.set_index("issue_id").loc[pick]
    a, b = st.columns([1.15, 1])
    with a, st.container(border=True):
        st.markdown(f"#### {LABELS.get(r.rule, r.rule).capitalize()}")
        st.markdown(f"**Bill / line:** {r.invoice_id}{' · line ' + r.line_no if r.line_no else ''} · **{r.contractor}** · {r.period}  \n"
                    f"**Source:** {r.source} · **Recommended:** {r.recommended.lower()} · **Flagged:** {money(r.flagged)}")
        st.markdown(f"**Evidence:** {r.detail}")
        st.markdown(f"**Suggested next step:** {ACTIONS.get(r.rule, '')}")
        st.markdown(f"**Status:** {r.status}" + (f" · owner {r.owner}" if r.owner else "") + (f" · due {r.due}" if r.due else "")
                    + (f"  \n**Decision:** {r.decision} · reason: {r.reason} · adjustment {money(r.confirmed or 0)}"
                       + (" · **override**" if r.override else "") if r.decision else ""))
    with b, st.container(border=True):
        st.markdown("#### Record an action")
        nxt = D.next_statuses(r.status)
        if not nxt:
            st.success("Closed.")
        else:
            to = st.radio("Next step", nxt, horizontal=True, key=f"to_{pick}",
                          format_func=lambda s: "Reopen" if s == "Investigating" and r.status in ("Decided", "Reconciled", "Closed") else s)
            with st.form(f"act_{pick}_{to}"):
                user = st.text_input("Your name and role", placeholder="e.g. M. Kumar (program manager)")
                o, due, dec, amt, reason = None, None, None, None, None
                if to in ("Assigned", "Investigating"):
                    o = st.text_input("Owner", r.owner or "")
                    due = st.date_input("Due", date.today() + timedelta(days=7))
                if to == "Decided":
                    dec = st.selectbox("Decision", D.DECISIONS)
                    amt = st.number_input("Confirmed adjustment, $ (used when the decision is 'Adjust')", 0.0,
                                          value=float(r.flagged), step=100.0, format="%.2f")
                    reason = st.selectbox("Reason code", D.REASONS)
                hint = {"Assigned": "Who looks at this, and by when.", "Investigating": "What you are asking for or checking.",
                        "Decided": "What you checked and why you decided this.", "Reconciled": "Which payment the adjustment went into.",
                        "Closed": "Anything left to note."}.get(to, "")
                note = st.text_area("Note (required)", placeholder=hint)
                if st.form_submit_button("Save to the record", type="primary"):
                    try:
                        D.act(pick, user, to, note, owner=o or None, due=due, decision=dec, confirmed=amt, reason=reason)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            st.caption("Certifying a recommended hold as billed is recorded as an override and needs a written reason.")
    st.markdown("##### History")
    st.dataframe(D.history(pick), hide_index=True, width="stretch", column_config={
        "ts": "When", "user": "Who", "action": "Action", "from_status": "From", "to_status": "To", "note": "Note"})
    ev = D.events(prog)
    st.download_button("Download the full decision log (CSV)", ev.to_csv(index=False), f"decision_log_{prog}.csv", "text/csv")
    st.caption("Demo storage: decisions live in a small database inside the app and reset when the hosted app restarts. "
               "Download the log to keep it. A real deployment would use a managed database with sign-in.")
