"""Front page: the decision first. The largest line answers 'what do I need to do, and how much is at stake?'"""
import html

import pandas as pd
import streamlit as st

import decisions as D
from insights import money
from ui.payapp_pages import ACTIONS, LABELS, findings

CSS = """
<style>
.fp-context { color: var(--muted); font-size: .92rem; letter-spacing: .02em; margin-bottom: .6rem; }
.fp-head { font-size: 3.1rem; line-height: 1.1; font-weight: 500; letter-spacing: -0.02em; color: var(--ink); margin: 0 0 .9rem; max-width: 1050px; }
.fp-lede { font-size: 1.18rem; line-height: 1.5; color: var(--ink2); max-width: 900px; margin-bottom: 1.6rem; }
.fp-band { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 28px; padding: 34px 38px; border-radius: 28px;
  background: radial-gradient(circle at 82% 20%, rgba(194,219,255,.55), rgba(194,219,255,0) 40%), linear-gradient(120deg, #0b57d0 0%, #1a73e8 55%, #4c8df6 100%);
  margin: .4rem 0 2.4rem; box-shadow: 0 6px 20px rgba(11,87,208,.22); }
.fp-band .k { font-size: .95rem; color: rgba(255,255,255,.92); font-weight: 500; }
.fp-band .v { font-size: 2.9rem; font-weight: 500; letter-spacing: -0.02em; color: #fff; margin-top: 4px; }
.fp-band .s { font-size: .9rem; color: rgba(255,255,255,.88); margin-top: 2px; }
.fp-h2 { font-size: 1.8rem; font-weight: 500; color: var(--ink); margin: 1.2rem 0 .2rem; }
.fp-sub { color: var(--muted); font-size: .98rem; margin-bottom: .8rem; }
.fp-row { display: flex; flex-wrap: wrap; gap: 18px; align-items: flex-start; padding: 18px 20px; margin-bottom: 10px; background: #fff; border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 1px 2px rgba(60,64,67,.06); }
.fp-ico { flex: 0 0 30px; display: flex; justify-content: center; padding-top: 1px; }
.fp-legend { display: flex; flex-wrap: wrap; gap: 18px; align-items: center; color: var(--muted); font-size: .85rem; margin: -.2rem 0 .6rem; }
.fp-legend span { display: inline-flex; align-items: center; gap: 6px; } .fp-legend svg { width: 16px; height: 16px; }
.fp-main { flex: 1 1 420px; min-width: 0; } .fp-title { font-size: 1.08rem; color: var(--ink); font-weight: 500; }
.fp-why { color: var(--ink2); font-size: .95rem; line-height: 1.5; margin-top: 4px; } .fp-meta { color: var(--muted); font-size: .82rem; margin-top: 6px; }
.fp-amt { flex: 0 0 auto; text-align: right; font-size: 1.45rem; font-weight: 600; color: var(--ink); }
.fp-bars { display: grid; grid-template-columns: minmax(170px, 260px) 1fr; gap: 12px 20px; align-items: center; font-size: .95rem; color: var(--ink2); max-width: 900px; }
.fp-bar { display: flex; align-items: center; gap: 12px; } .fp-bar i { display: block; height: 14px; border-radius: 0 4px 4px 0; background: #2a78d6; }
.fp-bar b { font-weight: 500; color: var(--ink); min-width: 72px; }
</style>
"""


ICON = {
    "hold": ('<svg width="26" height="26" viewBox="0 0 24 24" role="img" aria-label="Recommend hold"><title>Recommend hold</title>'
             '<path d="M8 2h8l6 6v8l-6 6H8l-6-6V8z" fill="#e5484d"/><rect x="7" y="10.5" width="10" height="3" rx="1" fill="#fff"/></svg>'),
    "review": ('<svg width="26" height="26" viewBox="0 0 24 24" role="img" aria-label="Needs review"><title>Needs review</title>'
               '<path d="M12 2.5 23 21.5H1z" fill="#f5b120"/><rect x="11" y="9" width="2" height="7" rx="1" fill="#1a1200"/>'
               '<circle cx="12" cy="18.4" r="1.2" fill="#1a1200"/></svg>'),
    "flag": ('<svg width="26" height="26" viewBox="0 0 24 24" role="img" aria-label="Red flag"><title>Red flag</title>'
             '<rect x="4" y="2" width="2" height="20" rx="1" fill="#5e6368"/><path d="M6 3h13l-3 4.5 3 4.5H6z" fill="#e5484d"/></svg>'),
}


def _h(x):
    return html.escape(str(x))


def page(ctx):
    st.markdown(CSS, unsafe_allow_html=True)
    g, t = ctx.gates.copy(), ctx.tables
    for c in ["gross_amount", "amount_at_risk"]:
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    month = g.period.max()
    gm = g[g.period == month]
    held = gm[gm.gate == "HOLD"]
    billed, flagged_m = gm.gross_amount.sum(), gm.amount_at_risk.sum()
    safe = gm[gm.gate == "PASS"].gross_amount.sum()
    mlabel = pd.Period(month, "M").strftime("%B %Y")
    asof = pd.to_datetime(t["invoices"].invoice_date, errors="coerce").max()

    # red flags (override everything)
    reds = []
    if hasattr(ctx, "health") and "red_flags" in ctx.health:
        for r in ctx.pt.merge(ctx.health, on="project_id").itertuples():
            for f in (r.red_flags or []):
                sev = (getattr(r, "delay_weeks", 0) or 0) if "weeks" in f else 0
                reds.append((ctx.labels.get(r.project_id, r.project_id), f, sev))
        reds.sort(key=lambda x: -x[2])

    # the headline: the single most important true sentence today
    if reds:
        head = f"{reds[0][0]}: {reds[0][1][0].lower()}{reds[0][1][1:]}."
        lede = (f"That overrides everything else. On {mlabel} bills: hold {money(held.amount_at_risk.sum())} on {len(held)} "
                f"bill{'s' if len(held) != 1 else ''}, pay {money(billed - flagged_m)}.")
    elif len(held):
        head = (f"Hold {money(held.amount_at_risk.sum())} on {len(held)} bill{'s' if len(held) != 1 else ''}. "
                f"Pay the other {money(billed - flagged_m)}.")
        lede = (f"{len(gm) - len(held)} of {len(gm)} {mlabel} bills match the contract and the site records. "
                f"The held amounts are recommendations: a reviewer decides each one.")
    else:
        head = f"All {len(gm)} {mlabel} bills pass. Safe to pay {money(billed)}."
        lede = "Every line matches the contract, the approvals and what the site team verified."

    st.markdown(f'<div class="fp-context">{_h(ctx.program_label)} · {mlabel} bills · data as of '
                f'{asof:%d %b %Y}</div><div class="fp-head">{_h(head)}</div><div class="fp-lede">{_h(lede)}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="fp-band"><div><div class="k">Billed in {mlabel}</div><div class="v">{money(billed)}</div>'
        f'<div class="s">{len(gm)} bills</div></div>'
        f'<div><div class="k">Flagged</div><div class="v">{money(flagged_m)}</div><div class="s">{len(held)} recommended hold{"s" if len(held) != 1 else ""} · '
        f'{int((gm.gate == "REVIEW").sum())} to review</div></div>'
        f'<div><div class="k">Safe to pay now</div><div class="v">{money(billed - flagged_m)}</div>'
        f'<div class="s">{int((gm.gate == "PASS").sum())} bills pass every check</div></div></div>', unsafe_allow_html=True)

    if reds:
        st.markdown('<div class="fp-h2">Red flags</div><div class="fp-sub">These override the health score.</div>',
                    unsafe_allow_html=True)
        st.markdown("".join(f'<div class="fp-row"><span class="fp-ico">{ICON["flag"]}</span><div class="fp-main">'
                            f'<div class="fp-title">{_h(n)}</div><div class="fp-why">{_h(f)}</div></div></div>' for n, f, _ in reds),
                    unsafe_allow_html=True)

    # decision queue: open issues, largest first
    prog = ctx.source_key
    D.sync(prog, findings(ctx))
    iss = D.issues(prog)
    open_ = iss[iss.status.isin(D.OPEN)].sort_values("flagged", ascending=False)
    st.markdown(f'<div class="fp-h2">Decisions waiting</div><div class="fp-sub">{len(open_)} open issues across bills, pay '
                f'applications and cost-plus checks. Largest first; each says why and what to do next.</div>'
                f'<div class="fp-legend"><span>{ICON["hold"]} recommend hold</span><span>{ICON["review"]} needs review</span></div>',
                unsafe_allow_html=True)
    rows = []
    for r in open_.head(6).itertuples():
        chip = "hold" if r.recommended == "HOLD" else "review"
        meta = f"{r.status}" + (f" · {r.owner}" if r.owner else " · no owner yet") + (f" · due {r.due}" if r.due else "")
        amt = money(r.flagged) if r.flagged else '<span style="font-size:.9rem;color:var(--muted)">document needed</span>'
        rows.append(f'<div class="fp-row"><span class="fp-ico">{ICON[chip]}</span>'
                    f'<div class="fp-main"><div class="fp-title">{_h(r.contractor)} · {_h(r.invoice_id)} · '
                    f'{_h(LABELS.get(r.rule, r.rule))}</div><div class="fp-why">{_h(ACTIONS.get(r.rule, ""))}</div>'
                    f'<div class="fp-meta">{_h(meta)} · {_h(r.source)}</div></div><div class="fp-amt">{amt}</div></div>')
    if rows:
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.button(f"Open all {len(open_)} in Issues & decisions →", type="tertiary", on_click=ctx.go, args=("issues",))
    else:
        st.success("No open issues.")

    # why money is held this month
    ex = ctx.exceptions.merge(g[["invoice_id", "period"]], on="invoice_id")
    by = ex[ex.period == month].groupby("rule").amount_at_risk.sum().sort_values(ascending=False)
    by = by[by > 0]
    if len(by):
        top = by.iloc[0]
        share = top / by.sum()
        st.markdown(f'<div class="fp-h2">Why money is flagged in {mlabel}</div><div class="fp-sub">'
                    f'{_h(LABELS.get(by.index[0], by.index[0]).capitalize())} is {share:.0%} of it.</div>', unsafe_allow_html=True)
        st.markdown('<div class="fp-bars">' + "".join(
            f'<div>{_h(LABELS.get(k, k).capitalize())}</div><div class="fp-bar"><i style="width:{max(100 * v / top, 0.6):.1f}%"></i>'
            f'<b>{money(v)}</b></div>' for k, v in by.items()) + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="fp-sub" style="margin-top:2.2rem">All data is synthetic. The globe and program cards are on '
                '<b>Mission Control</b>; the full checks on <b>Program analytics</b>.</div>', unsafe_allow_html=True)
