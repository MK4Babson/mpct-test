"""
Change orders: should we approve, negotiate or reject, and is the change justified and fairly priced?

  F10  decision tree   expected cost of Approve / Negotiate / Reject, with dominance and sensitivity
  F11  analyst         text signals (Naive Bayes on justification text), price premium vs base rate, base-scope check,
                       timing, then a recommendation and a memo (template, or Gemini when a key is set)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CO_TEXT = ROOT / "data" / "training" / "co_text.csv"

DEFAULTS = dict(p_negotiate=0.6, settle_frac=0.85, claim_mult=1.25, legal_cost=25000.0, day_cost=40000.0,
                base_claim=0.15, claim_slope=0.70, negotiate_days=10, schedule_share=0.3)


# ---------------------------------------------------------------- F11 text model
def train_text_model():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
    df = pd.read_csv(CO_TEXT)
    mk = lambda: CountVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")  # noqa: E731
    pipe = make_pipeline(mk(), CalibratedClassifierCV(MultinomialNB(alpha=1.0), method="sigmoid", cv=5))
    cv = cross_val_score(pipe, df.text, df.justified, cv=5, scoring="accuracy")
    pipe.fit(df.text, df.justified)
    vec = pipe.named_steps["countvectorizer"]
    nb = MultinomialNB(alpha=1.0).fit(vec.transform(df.text), df.justified)  # uncalibrated copy, for the word weights
    lp = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    words = pd.DataFrame({"term": vec.get_feature_names_out(), "log_odds_justified": lp})
    return {"pipe": pipe, "cv_accuracy": float(cv.mean()), "cv_sd": float(cv.std()), "words": words, "n": len(df),
            "base_rate": float(df.justified.mean())}


def text_signals(model, text, top=4):
    vec = model["pipe"].named_steps["countvectorizer"]
    p = float(model["pipe"].predict_proba([text])[0, 1])
    present = set(vec.build_analyzer()(text))
    w = model["words"][model["words"].term.isin(present)].sort_values("log_odds_justified")
    neg = w.head(top)[w.head(top).log_odds_justified < 0].term.tolist()
    pos = w.tail(top)[w.tail(top).log_odds_justified > 0].term.tolist()[::-1]
    return p, pos, neg


# ---------------------------------------------------------------- F10 decision tree
def decision(amount, p_justified, delay_days, a=None):
    a = {**DEFAULTS, **(a or {})}
    q = min(0.95, a["base_claim"] + a["claim_slope"] * p_justified)  # chance the contractor wins/claims if we reject
    # negotiating takes time; if the change is on the critical path, part of that time delays the job
    neg_delay = a["negotiate_days"] * a["schedule_share"] * a["day_cost"] if delay_days > 0 else 0.0
    claim_cost = amount * a["claim_mult"] + a["legal_cost"] + delay_days * a["day_cost"]
    branches = {
        "Approve as priced": [(1.0, amount, "Pay the quoted amount")],
        "Negotiate": [(a["p_negotiate"], amount * a["settle_frac"] + neg_delay, f"Settle at {a['settle_frac']:.0%} of quote"),
                      (1 - a["p_negotiate"], amount + neg_delay, "No movement: approve at quote")],
        "Reject": [(q, claim_cost, "Contractor claims and wins (claim premium + legal + delay)"),
                   (1 - q, 0.0, "Contractor absorbs it")],
    }
    ev = {k: sum(p * c for p, c, _ in v) for k, v in branches.items()}
    worst = {k: max(c for _, c, _ in v) for k, v in branches.items()}
    best = min(ev, key=ev.get)
    dominated = [k for k in branches if k != "Negotiate" and ev["Negotiate"] <= ev[k] and worst["Negotiate"] <= worst[k]]
    return {"branches": branches, "ev": ev, "worst": worst, "best": best, "q_claim": q, "claim_cost": claim_cost,
            "dominated": dominated}


def breakeven_p(amount, delay_days, a=None):
    """Smallest P(justified) at which Reject stops being cheapest (sensitivity)."""
    for p in np.linspace(0, 1, 201):
        d = decision(amount, p, delay_days, a)
        if d["best"] != "Reject":
            return float(p)
    return 1.0


def tree_dot(d, amount):
    fmt = lambda x: f"${x/1e3:,.0f}K" if abs(x) < 1e6 else f"${x/1e6:,.2f}M"  # noqa: E731
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [fontname="Helvetica", fontsize=11, color="#4fd1ff", fontcolor="#e8eef8", style=rounded];',
             'edge [color="#8792a6", fontcolor="#b9c3d3", fontsize=10];',
             f'root [shape=box, label="Change order\\n{fmt(amount)}", color="#4fd1ff"];']
    for i, (opt, br) in enumerate(d["branches"].items()):
        best = opt == d["best"]
        col = "#0ca30c" if best else "#8792a6"
        lines.append(f'o{i} [shape=circle, label="{fmt(d["ev"][opt])}", color="{col}", penwidth={3 if best else 1}];')
        lines.append(f'root -> o{i} [label="{opt}{"  ✓" if best else ""}", color="{col}", penwidth={2.5 if best else 1}];')
        for j, (p, c, lab) in enumerate(br):
            lines.append(f'l{i}{j} [shape=box, label="{lab}\\n{fmt(c)}", color="#2b3a58"];')
            lines.append(f'o{i} -> l{i}{j} [label="p = {p:.2f}"];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- analyst table
def analyze(tables, details, model, a=None, should=None, delay_fn=None):
    """delay_fn(project_id, base_item_code, claimed_days) -> (days, note): optional schedule-based delay (time impact analysis)."""
    co = tables["change_orders"].copy()
    po = tables["purchase_orders"].set_index("po_id")
    pol = tables["po_lines"]
    lines = tables["invoice_lines"]
    inv = tables["invoices"]
    if "project_id" not in co.columns:
        co["project_id"] = co.po_id.map(po.project_id)
    if "prime_id" not in co.columns:
        co["prime_id"] = co.po_id.map(po.prime_id)
    co = co.merge(details, on="co_id", how="left") if details is not None and len(details) else co.assign(
        justification=co.get("description", ""), schedule_days_if_rejected=0, cause_category=None)
    co["justification"] = co.justification.fillna(co.get("description", "").astype(str) if "description" in co else "")
    base = pol[pol.line_source == "CONTRACT"].set_index(["po_id", "item_code"])
    billed = (lines[lines.line_type == "SOV"].merge(inv[["invoice_id", "po_id"]], on="invoice_id")
              .groupby(["po_id", "ref_id"]).qty.sum())
    rows = []
    for r in co.itertuples():
        p, pos, neg = text_signals(model, str(r.justification))
        key = (r.po_id, getattr(r, "base_item_code", None))
        base_rate = base.contract_unit_rate.get(key, np.nan) if key[1] is not None and key in base.index else np.nan
        base_qty = base.contract_qty.get(key, np.nan) if key[1] is not None and key in base.index else np.nan
        premium = 100 * (r.co_unit_rate / base_rate - 1) if pd.notna(base_rate) and base_rate else np.nan
        used = billed.get(key, 0.0) / base_qty * 100 if pd.notna(base_qty) and base_qty else np.nan
        sc_hi = np.nan
        if should is not None and len(should) and key[1] is not None:
            m = should[(should.po_id == r.po_id) & (should.item_code == key[1])]
            sc_hi = float(m.pi_hi.iloc[0]) if len(m) else np.nan
        claimed = int(getattr(r, "schedule_days_if_rejected", 0) or 0)
        days, delay_note = claimed, "as claimed"
        if delay_fn is not None and claimed > 0:
            days, delay_note = delay_fn(r.project_id, key[1], claimed)
        d = decision(float(r.co_amount), p, days, a)
        concerns = []
        if p < 0.4:
            concerns.append(f"wording reads like a contractor-side cause ({', '.join(neg[:3]) or 'weak justification'})")
        if pd.notna(premium) and premium > 3:
            concerns.append(f"unit rate {premium:.1f}% above the base contract rate")
        if pd.notna(sc_hi) and r.co_unit_rate > sc_hi:
            concerns.append("rate above the should-cost range")
        if pd.notna(used) and used < 80:
            concerns.append(f"only {used:.0f}% of the base quantity billed so far: the base scope may still cover it")
        rows.append({"co_id": r.co_id, "project_id": r.project_id, "prime_id": r.prime_id, "status": r.status,
                     "amount": float(r.co_amount), "justification": r.justification, "cause_category": getattr(r, "cause_category", None),
                     "p_justified": p, "positive_terms": ", ".join(pos), "negative_terms": ", ".join(neg),
                     "rate_premium_pct": premium, "base_used_pct": used, "should_cost_hi": sc_hi, "delay_days": days, "claimed_delay_days": claimed, "delay_basis": delay_note,
                     "ev_approve": d["ev"]["Approve as priced"], "ev_negotiate": d["ev"]["Negotiate"], "ev_reject": d["ev"]["Reject"],
                     "recommendation": d["best"], "expected_saving": d["ev"]["Approve as priced"] - d["ev"][d["best"]],
                     "concerns": "; ".join(concerns) if concerns else "none found"})
    return pd.DataFrame(rows)


def memo_prompt(row, project, contractor):
    return (f"You are a construction commercial manager at a data-center owner. Write a short, factual memo (under 180 words) "
            f"recommending what to do with change order {row['co_id']} on {project} from {contractor}. Amount ${row['amount']:,.0f}. "
            f"Justification text: \"{row['justification']}\". Model probability the change is owner-caused/justified: {row['p_justified']:.0%}. "
            f"Rate premium vs base contract: {row['rate_premium_pct'] if pd.notna(row['rate_premium_pct']) else 'n/a'}%. "
            f"Base quantity billed so far: {row['base_used_pct'] if pd.notna(row['base_used_pct']) else 'n/a'}%. "
            f"Concerns: {row['concerns']}. Decision-tree expected cost: approve ${row['ev_approve']:,.0f}, negotiate ${row['ev_negotiate']:,.0f}, "
            f"reject ${row['ev_reject']:,.0f}; recommended: {row['recommendation']}. Include: recommendation, 2-3 reasons, documents to request, "
            f"and a reminder that the project manager makes the final decision. Do not invent facts.")


def memo_template(row, project, contractor):
    docs = []
    if pd.notna(row["base_used_pct"]) and row["base_used_pct"] < 80:
        docs.append("a quantity take-off showing why the base contract quantity is not enough")
    if pd.notna(row["rate_premium_pct"]) and row["rate_premium_pct"] > 3:
        docs.append("a cost breakdown supporting the unit rate above the base contract rate")
    if row["p_justified"] < 0.5:
        docs.append("the owner directive, RFI or drawing revision that caused the change")
    docs = docs or ["the signed field directive and backup quantities"]
    return (f"**Change order {row['co_id']}: {project}, {contractor}**\n\n"
            f"**Recommendation: {row['recommendation']}** (expected cost ${row[{'Approve as priced': 'ev_approve', 'Negotiate': 'ev_negotiate', 'Reject': 'ev_reject'}[row['recommendation']]]:,.0f} "
            f"vs ${row['amount']:,.0f} quoted).\n\n"
            f"- Justification reads as owner-caused with probability {row['p_justified']:.0%}"
            f"{' (signals: ' + row['positive_terms'] + ')' if row['positive_terms'] else ''}"
            f"{'; contractor-side signals: ' + row['negative_terms'] if row['negative_terms'] else ''}.\n"
            f"- Concerns: {row['concerns']}.\n"
            f"- If rejected, the expected schedule impact is {row['delay_days']} days.\n\n"
            f"Request before deciding: {'; '.join(docs)}.\n\n"
            f"*Drafted automatically from the data. The project manager makes the final decision.*")
