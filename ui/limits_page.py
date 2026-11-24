"""'What this is not (yet)': every known limit in one place, and the path to a real pilot."""
import streamlit as st

LIMITS = [
    ("Data", [
        "All data is synthetic. No real contractor bill has been run through the engine yet.",
        "Accuracy scores use errors planted by the same author who wrote the checks, so a perfect score proves the code runs, "
        "not that it works on real bills.",
    ]),
    ("Contracts and payments", [
        "West Texas now has a G702/G703-style ledger, stored materials, lien waivers, insurance and cost-plus (GMP) checks on "
        "each prime's general conditions. The main work packages are still unit-rate and lump-sum; a full GMP prime contract "
        "(cost of work plus fee under a maximum) is not modelled end to end.",
        "Not yet checked: payment and performance bonds, allowances, retainage release at substantial completion, and "
        "state-specific lien-waiver forms. The canal program has none of the US pay-application layer (it uses Indian running bills).",
        "A hold is a recommendation. The contract, the evidence and an authorized reviewer decide whether payment is withheld, and "
        "most US states require written notice with reasons.",
    ]),
    ("Decisions and evidence", [
        "Confirmed adjustments and money saved come only from decisions recorded on the Issues & decisions page. The demo "
        "decisions there are invented, so those two numbers prove the workflow, not real savings.",
        "Decisions are stored in a small demo database that resets when the hosted app restarts, and the name on each action is "
        "typed, not signed in. Findings from different checks can overlap on the same dollars.",
        "Not yet: contractor responses inside the app, escalation deadlines, and posting certified amounts back to an ERP.",
        "A delivery photo is supporting evidence, never proof. Source documents are not versioned, so a past decision cannot yet "
        "be rebuilt exactly.",
    ]),
    ("Schedule", [
        "Each building uses a template task network with illustrative durations, not the contractors' Primavera P6 schedules.",
        "No integrated master schedule and no cross-prime handovers, which is where most owner schedule risk sits.",
        "The Monte Carlo varies tasks independently (no correlation), and the delay-claim tab is screening only, not forensic "
        "delay analysis.",
        "Verified quantity is used as progress. Real earned progress often needs milestone rules, especially for testing and "
        "commissioning.",
    ]),
    ("Cost models", [
        "Alpha (growth beyond the market) is a residual, not proof of poor contractor performance.",
        "Should-cost is a screening band, not an enforceable fair price.",
        "The change-order tree is scenario analysis. Entitlement under the contract comes first.",
    ]),
    ("Materials", [
        "Demand comes from the schedule forecast, not from approved engineering quantities.",
        "No reservations, quarantine, serial numbers or in-transit stock. EOQ and safety stock do not suit one-off engineered "
        "equipment, which needs milestone tracking instead.",
        "Transfers minimize freight only. They do not yet price the schedule risk of leaving a site short.",
    ]),
    ("Labs", [
        "Machine learning, the macroeconomic models, the association and lead-lag tests, carbon and the BigQuery package are "
        "method demonstrations. Their outputs are not validated on real projects.",
        "The useful macro parts are exposure-based: currency exposure on open orders, tariff and landed cost, commodity "
        "escalation. The central-bank and balance-of-payments tabs explain context; they do not predict package costs.",
    ]),
    ("Security and hosting", [
        "Hosted on Streamlit Community Cloud as a demo. No sign-in, no single sign-on, no audit logging, no data controls.",
        "The role selector is a view filter, not access control.",
        "No link to ERP or payment systems, so there is no reconciliation against the certified payment ledger.",
    ]),
]

PILOT = [
    ("Scope", "One program, two or three prime contractors, six months of past pay applications."),
    ("Mode", "Shadow mode: the app produces findings; the existing approval process stays in charge."),
    ("Ledger", "Rebuild every previously certified payment exactly from source records."),
    ("Exceptions", "Precision and recall by error type, judged by an independent reviewer, on a held-back period."),
    ("Productivity", "Confirmed dollars found per 100 reviewer hours, against the current process. Target 30% less effort."),
    ("Traceability", "Every material finding traced to its source documents, rule version and reviewer decision."),
]


def page(ctx):
    st.markdown('<div class="eyebrow">Honest limits</div><div class="hero-title">What this is not (yet)</div>'
                '<div class="hero-sub">This app is an analytical prototype on synthetic data. It is not a validated system for '
                'live contractor payments. Every known limit is listed here, followed by what a real pilot would need.</div>',
                unsafe_allow_html=True)
    for title, items in LIMITS:
        st.markdown(f"#### {title}")
        st.markdown("\n".join(f"- {x}" for x in items))
    st.markdown("#### What a real pilot would test")
    st.dataframe([{"Test": a, "What it should show": b} for a, b in PILOT], hide_index=True, width="stretch")
    st.caption("The production path: keep the existing systems (Procore, Oracle, P6, ERP) and run this as an independent "
               "cross-prime assurance layer on top of them, after a security and data-governance review.")


def lab_banner():
    st.info("**Labs.** A method demonstration on synthetic data. Useful for exploring ideas; not validated for real decisions. "
            "See *What this is not (yet)*.", icon="⚗️")
