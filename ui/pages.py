"""Mission Control (home), Insights (project cards) and Project (deep dive) pages."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from insights import RULE_LABEL, money
from health import WEIGHTS, CASH_RULES
from ui import visuals
from ui.theme import SERIES, STATUS, GATE_COLOR, ACCENT

PILL = {"Healthy": "good", "Watch": "warn", "At risk": "crit", "No data": "warn"}
HCOL = {"Healthy": STATUS["good"], "Watch": STATUS["warning"], "At risk": STATUS["critical"], "No data": "#8792a6"}
FLAG_ICON = {"crit": "✕", "warn": "!", "info": "i"}
FLAG_COL = {"crit": "#ff6b6b", "warn": "#fab219", "info": "#7fb8ff"}
STATUS_TO_PIN = {"Healthy": "Green", "Watch": "Amber", "At risk": "Red"}


def ring(score, size=74, label=True):
    s = 0 if pd.isna(score) else max(0, min(100, score))
    col = HCOL["Healthy"] if s >= 80 else HCOL["Watch"] if s >= 60 else HCOL["At risk"]
    r = size / 2 - 6
    circ = 2 * np.pi * r
    txt = "–" if pd.isna(score) else f"{s:.0f}"
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" stroke="rgba(255,255,255,0.08)" stroke-width="6" fill="none"/>'
            f'<circle cx="{size/2}" cy="{size/2}" r="{r}" stroke="{col}" stroke-width="6" fill="none" stroke-linecap="round" '
            f'stroke-dasharray="{circ * s / 100:.1f} {circ:.1f}" transform="rotate(-90 {size/2} {size/2})" '
            f'style="filter:drop-shadow(0 0 6px {col}88)"/>'
            f'<text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle" fill="#e8eef8" '
            f'font-family="JetBrains Mono, monospace" font-size="{size*0.28:.0f}" font-weight="600">{txt}</text>'
            + (f'<text x="50%" y="74%" text-anchor="middle" fill="#8792a6" font-size="{size*0.11:.0f}" font-family="Space Grotesk, system-ui, sans-serif" letter-spacing="1">HEALTH</text>' if label else "")
            + "</svg>")


def subbars(h):
    out = '<div class="grid4">'
    for k, lab in [("cost", "Cost"), ("schedule", "Schedule"), ("billing", "Billing"), ("cash", "Cash")]:
        v = h.get(f"{k}_score", np.nan)
        col = "#8792a6" if pd.isna(v) else (HCOL["Healthy"] if v >= 80 else HCOL["Watch"] if v >= 60 else HCOL["At risk"])
        w = 0 if pd.isna(v) else v
        out += (f'<div><div class="lbl"><span>{lab}</span><span class="kv">{"–" if pd.isna(v) else f"{v:.0f}"}</span></div>'
                f'<div class="subbar"><div style="width:{w:.0f}%;background:{col}"></div></div></div>')
    return out + "</div>"


def flags_html(flags, n=3):
    if not flags:
        return '<div class="flag" style="color:#3fd13f">✓ No open flags</div>'
    return "".join(f'<div class="flag"><span style="color:{FLAG_COL[l]};font-weight:700">{FLAG_ICON[l]}</span> {t}</div>'
                   for l, t in flags[:n])


HELP = {
    "programs": "Construction programs tracked here. A program is a group of related projects.",
    "sites": "Individual projects (buildings or canal divisions), each pinned at its coordinates.",
    "committed": "Total value of signed contracts plus approved change orders: what the owner has promised to pay.",
    "at_risk_sites": "Projects with a health score below 60. Act on these this week.",
    "program_health": "Average project health (0-100), weighted by contract value, so bigger projects count more. 80+ Healthy, 60-79 Watch, below 60 At risk. A red flag (finish 4+ weeks past contract, or 3%+ over budget) sets At risk whatever the score.",
    "vs_budget": "Approved budget minus forecast final cost (signed contracts + pending change orders). Positive = room left; negative = forecast overrun.",
    "verified": "Value of work measured on site as built, divided by contract value. The delta compares it with the schedule plan for the same month.",
    "blocked": "Money flagged by checks that recommend holding payment (e.g. duplicate, rate above contract). A recommendation: an authorized reviewer decides, under the contract.",
    "to_check": "Money on bills that failed a REVIEW check. Someone looks before paying, but payment isn't stopped.",
    "penalty": "Delay penalty (liquidated damages) the owner could claim if the forecast delay happens: weekly rate x weeks late, capped.",
    "budget": "Owner's approved budget for this contract, including contingency.",
    "forecast_cost": "Signed contract + approved changes + pending changes (assumes pending changes get approved: the cautious view).",
    "billed": "Value of work the contractor has billed so far (excluding duplicates).",
    "spi": "Schedule index = verified % / planned %. 1.00 = on plan; 0.88 = only 88% of the planned work done.",
    "finish": "Contract completion date vs the forecast: the planned finish moved by today's schedule slip (earned schedule), or given in the data. Ranges in Forecast & risk.",
    "first_pass": "Share of this project's bills that passed every check the first time. Low = the contractor's billing process needs fixing.",
    "pending": "Change orders submitted but not yet decided. They are included in the forecast cost.",
    "recover": "Flagged money that was already paid before the checks ran. Deduct it from the contractor's next bill.",
}


def back(ctx, target, label):
    st.button(label, key=f"back_{target}_{label}", on_click=ctx.go, args=(target, None, None), type="tertiary")


def fmt_date(d):
    return d.strftime("%d %b %Y") if pd.notna(d) else "–"


# ---------------------------------------------------------------- Mission Control
def home(ctx):
    st.markdown('<div class="eyebrow">Multi-Prime Commercial Control Tower</div>'
                '<div class="hero-title">Mission Control</div>'
                '<div class="hero-sub">Every active construction program, from orbit to invoice line. Pins mark each '
                'project\'s exact coordinates; color shows its health.</div>', unsafe_allow_html=True)
    snaps = ctx.snapshots
    pts, tour, i = [], [], 0
    for prog in snaps:
        ps = prog["projects"]
        for r in ps.itertuples():
            pts.append({"id": r.project_id, "i": i, "name": r.name, "lat": float(r.latitude), "lng": float(r.longitude),
                        "status": STATUS_TO_PIN.get(r.health_status, "Blue"), "location": r.location})
            i += 1
        lat, lng = float(ps.latitude.mean()), float(ps.longitude.mean())
        span = max(ps.latitude.max() - ps.latitude.min(), ps.longitude.max() - ps.longitude.min())
        tour.append({"lat": lat, "lng": lng, "alt": 0.9, "rotate": True, "ids": [], "program": prog["label"],
                     "title": prog["region"], "coord": f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}, {abs(lng):.4f}° {'E' if lng >= 0 else 'W'}",
                     "status": f"{len(ps)} projects · health {prog['health']:.0f}", "color": HCOL[prog["status"]]})
        for r in ps.itertuples():
            tour.append({"lat": float(r.latitude), "lng": float(r.longitude), "alt": max(0.16, span * 0.02), "rotate": False,
                         "ids": [r.project_id], "program": prog["label"], "title": f"{r.name}: {r.location}",
                         "coord": f"{abs(r.latitude):.4f}° {'N' if r.latitude >= 0 else 'S'}, {abs(r.longitude):.4f}° {'E' if r.longitude >= 0 else 'W'}",
                         "status": f"{r.health_status} · {r.health:.0f}" if pd.notna(r.health) else "No data",
                         "color": HCOL.get(r.health_status, ACCENT)})
    visuals.globe(pts, tour, height=540)

    tot_value = sum(p["committed"] for p in snaps)
    c = st.columns(4)
    c[0].metric("Programs", len(snaps), help=HELP["programs"])
    c[1].metric("Active sites", sum(len(p["projects"]) for p in snaps), help=HELP["sites"])
    c[2].metric("Committed value", money(tot_value), help=HELP["committed"])
    c[3].metric("Sites at risk", sum(int((p["projects"].health_status == "At risk").sum()) for p in snaps), help=HELP["at_risk_sites"])

    st.markdown("### Programs")
    cols = st.columns(len(snaps))
    for col, prog in zip(cols, snaps):
        with col, st.container(border=True):
            visuals.structure(prog["structure"], prog["progress"], height=150, cell=4, seed=hash(prog["label"]) % 997,
                              label=f'{visuals.STRUCT_LABEL.get(prog["structure"], "")} program')
            st.markdown(
                f'<div style="display:flex;gap:14px;align-items:center">{ring(prog["health"], 80)}'
                f'<div><p class="card-title">{prog["label"]}</p><p class="card-sub">{prog["region"]}</p>'
                f'<span class="pill {PILL[prog["status"]]}">{prog["status"]}</span></div></div>'
                f'<div class="stat-row"><div class="stat" title="{HELP["sites"]}"><div class="v">{len(prog["projects"])}</div><div class="k">Projects ⓘ</div></div>'
                f'<div class="stat" title="{HELP["committed"]}"><div class="v">{money(prog["committed"])}</div><div class="k">Committed ⓘ</div></div>'
                f'<div class="stat" title="{HELP["verified"]}"><div class="v">{prog["progress"]*100:.0f}%</div><div class="k">Verified ⓘ</div></div>'
                f'<div class="stat" title="{HELP["blocked"]}"><div class="v">{money(prog["blocked"])}</div><div class="k">Flagged ⓘ</div></div></div>'
                + flags_html(prog["flags"], 2), unsafe_allow_html=True)
            st.button("Open program insights →", key=f"open_{prog['source']}", width="stretch",
                      on_click=ctx.go, args=("insights", prog["source"], None))


# ---------------------------------------------------------------- Insights
def insights(ctx):
    pt, H = ctx.pt, ctx.health
    back(ctx, "home", "← Mission Control")
    st.markdown(f'<div class="eyebrow">Insights · {ctx.program_label}</div><div class="hero-title">Project health</div>'
                f'<div class="hero-sub">{ctx.region}. One card per project: health score, what drives it, and what needs '
                f'attention. Open a project for the full analysis.</div>', unsafe_allow_html=True)
    j = pt.merge(H, on="project_id")
    reds = [(ctx.labels.get(r.project_id, r.project_id), f) for r in j.itertuples() for f in (r.red_flags or [])]
    if reds:
        st.error("**Red flags: these need attention whatever the health score says.**  \n"
                 + "  \n".join(f"**{n}:** {f}" for n, f in reds), icon="🚩")
    else:
        st.success("No red flags: no project is forecast more than 4 weeks past its contract date or more than 3% over budget. "
                   "A red flag would override the health score.", icon=":material/check_circle:")
    w = j.committed.where(j.health.notna(), 0)
    prog_health = (j.health.fillna(0) * w).sum() / w.sum() if w.sum() else np.nan
    k = st.columns(6)
    k[0].metric("Program health", f"{prog_health:.0f}/100" if pd.notna(prog_health) else "–", "value-weighted", delta_color="off", delta_arrow="off", help=HELP["program_health"])
    k[1].metric("Committed", money(j.committed.sum()), help=HELP["committed"])
    if ctx.has_ctl:
        vac = j.vac.sum()
        k[2].metric("Vs budget", money(abs(vac)), "under budget" if vac >= 0 else "over budget",
                    delta_color="normal" if vac >= 0 else "inverse", help=HELP["vs_budget"])
    else:
        k[2].metric("Vs budget", "–")
    ver = 100 * j.verified.sum() / j.committed.sum()
    plan = (j.planned_pct * j.committed).sum() / j.committed.sum() if j.planned_pct.notna().any() else np.nan
    k[3].metric("Verified", f"{ver:.0f}%", f"{ver - plan:+.0f} pts vs plan" if pd.notna(plan) else None, help=HELP["verified"])
    k[4].metric("Flagged: recommend hold", money(j.blocked.fillna(0).sum()), help=HELP["blocked"])
    k[5].metric("Penalty risk", money(j.ld_exposure.fillna(0).sum()) if ctx.has_ctl else "–", help=HELP["penalty"])
    notes = []
    if pd.notna(plan) and ver < plan - 0.5:
        gap_val = (plan - ver) / 100 * j.committed.sum()
        jj = j.assign(gap=(j.planned_pct - j.verified_pct) / 100 * j.committed).sort_values("gap", ascending=False)
        worst = [f"{n} ({g_ / 1e6:.1f}M)" for n, g_ in zip(jj.name, jj.gap) if g_ > 0.1 * gap_val][:3]
        notes.append(f"<b>Behind plan:</b> {plan - ver:.0f} pts ≈ <b>{money(gap_val)}</b> of work not yet built, mostly at {', '.join(worst)}.")
    if ctx.has_ctl:
        over = j[j.vac < 0]
        if len(over):
            notes.append(f"<b>Over budget:</b> {', '.join(over.name)} ({money(-over.vac.sum())}), even though the program total is "
                         f"{'under' if vac >= 0 else 'over'} budget.")
    if notes:
        st.markdown('<div class="howto">' + "<br>".join(notes) + "</div>", unsafe_allow_html=True)

    with st.expander("How the health score works"):
        st.markdown(
            f"""
| Part | Weight | How it's scored |
|---|---|---|
| **Cost** | {WEIGHTS['cost']:.0%} | Budget headroom: 3%+ under budget = 100, falling to 0 at 5% over budget |
| **Schedule** | {WEIGHTS['schedule']:.0%} | The worse of: schedule index (1.00 = 100, 0.70 = 0) and weeks late (−6 points per week) |
| **Billing** | {WEIGHTS['billing']:.0%} | 60% first-pass rate + 40% billing accuracy (−10 points per 1% of billing at risk) |
| **Cash** | {WEIGHTS['cash']:.0%} | 100 minus penalties: {'; '.join(f'{n} × {w:g}' for n, w in CASH_RULES)} |

**80+ Healthy · 60–79 Watch · below 60 At risk.** If a part has no data (e.g., no budget file), the other parts are re-weighted.

**Reading a card:** the ring is the overall score; the four bars show *which kind* of problem it is. Low **Cost** → review
pending change orders and budget; low **Schedule** → ask for a recovery plan; low **Billing** → meet the contractor's billing
team and tighten checks; low **Cash** → recover overpaid money, chase advance recovery, clear overdue bills. **Done vs Planned**
shows how far behind or ahead the work is; **vs budget** is the forecast surplus (+) or overrun (−); the flags list the specific
items to act on, most serious first.
""")
    order = j.sort_values("health", na_position="last")
    per_row = 3
    rows = [order.iloc[i:i + per_row] for i in range(0, len(order), per_row)]
    for chunk in rows:
        cols = st.columns(per_row)
        for col, r in zip(cols, chunk.itertuples()):
            with col, st.container(border=True):
                visuals.structure(ctx.structure_of(r.project_id), (r.verified_pct or 0) / 100, height=112, cell=3,
                                  seed=abs(hash(r.project_id)) % 997)
                hs = r.health_status
                st.markdown(
                    f'<div style="display:flex;gap:12px;align-items:center">{ring(r.health, 72)}'
                    f'<div style="flex:1"><p class="card-title">{r.name}</p><p class="card-sub">{r.contractor}<br>{ctx.location_of(r.project_id)}</p>'
                    f'<span class="pill {PILL[hs]}">{hs}</span></div></div>'
                    + subbars(r._asdict())
                    + f'<div class="stat-row"><div class="stat"><div class="v">{r.verified_pct:.0f}%</div><div class="k">Done</div></div>'
                    f'<div class="stat"><div class="v">{"–" if pd.isna(r.planned_pct) else f"{r.planned_pct:.0f}%"}</div><div class="k">Planned</div></div>'
                    f'<div class="stat"><div class="v">{"–" if pd.isna(r.vac) else ("+" if r.vac >= 0 else "−") + money(abs(r.vac))}</div><div class="k">vs budget</div></div>'
                    f'<div class="stat"><div class="v">{fmt_date(r.forecast_finish) if ctx.has_ctl else "–"}</div><div class="k">Forecast finish</div></div></div>'
                    + flags_html(r.alerts, 3), unsafe_allow_html=True)
                st.button("Open project →", key=f"proj_{r.project_id}", width="stretch",
                          on_click=ctx.go, args=("project", None, r.project_id))

    st.markdown("### Program decisions this week")
    for d in ctx.insights[:5]:
        col = {"critical": "#ff6b6b", "warning": "#fab219", "info": "#7fb8ff", "good": "#3fd13f"}[d["level"]]
        st.markdown(f'<div style="border-left:3px solid {col};padding:6px 12px;margin:6px 0;background:rgba(14,21,37,.5);border-radius:6px">'
                    f'<b>{d["title"]}</b><br><span class="muted">{d["action"]}</span></div>', unsafe_allow_html=True)
    st.button("Open full program analytics →", on_click=ctx.go, args=("analytics", None, None))


# ---------------------------------------------------------------- Project
def _period_end(p):
    return pd.Period(p, "M").end_time.normalize()


def project(ctx):
    pt, H = ctx.pt, ctx.health
    ids = list(pt.project_id)
    if not ids:
        st.info("No projects in the current selection.")
        return
    b1, b2, _ = st.columns([1, 1, 6])
    with b1:
        back(ctx, "insights", "← Insights")
    with b2:
        back(ctx, "home", "⌂ Mission Control")
    cur = st.session_state.get("project") if st.session_state.get("project") in ids else ids[0]
    pick = st.selectbox("Project", ids, index=ids.index(cur), format_func=lambda p: ctx.labels.get(p, p), label_visibility="collapsed")
    st.session_state["project"] = pick
    r = pt.set_index("project_id").loc[pick]
    h = H.set_index("project_id").loc[pick]
    lat, lng = ctx.coords_of(pick)

    st.markdown(f'<div class="eyebrow">Project · {ctx.program_label}</div><div class="hero-title">{r["name"]}</div>'
                f'<div class="hero-sub">{ctx.full_name_of(pick)} · {r.contractor} · {ctx.location_of(pick)}'
                + (f' · <span class="kv">{lat:.4f}°, {lng:.4f}°</span>' if pd.notna(lat) else "") + "</div>", unsafe_allow_html=True)

    a, b = st.columns([3, 2])
    with a:
        visuals.structure(ctx.structure_of(pick), (r.verified_pct or 0) / 100, height=250, cell=5,
                          seed=abs(hash(pick)) % 997, label=f"{visuals.STRUCT_LABEL.get(ctx.structure_of(pick), '')} · live build")
    with b:
        if pd.notna(lat):
            pin = [{"id": pick, "i": 0, "name": r["name"], "lat": lat, "lng": lng,
                    "status": STATUS_TO_PIN.get(h.health_status, "Blue"), "location": ctx.location_of(pick)}]
            stop = [{"lat": lat, "lng": lng, "alt": 0.22, "rotate": False, "ids": [pick], "program": ctx.program_label,
                     "title": r["name"], "coord": f"{lat:.4f}°, {lng:.4f}°", "status": h.health_status, "color": HCOL[h.health_status]}]
            visuals.globe(pin, stop, height=248)

    # health + KPIs
    c1, c2 = st.columns([1, 3])
    with c1, st.container(border=True):
        st.markdown(f'<div style="display:flex;justify-content:center">{ring(h.health, 128)}</div>'
                    f'<div style="text-align:center;margin:4px 0 8px"><span class="pill {PILL[h.health_status]}">{h.health_status}</span></div>'
                    + subbars(h.to_dict()) + flags_html(h["alerts"], 6), unsafe_allow_html=True)
    with c2:
        k = st.columns(4)
        k[0].metric("Budget", money(r.budget) if ctx.has_ctl else "–", help=HELP["budget"])
        k[1].metric("Forecast cost", money(r.eac), (f"{money(abs(r.vac))} {'under' if r.vac >= 0 else 'over'}" if ctx.has_ctl else None),
                    delta_color="normal" if (ctx.has_ctl and r.vac >= 0) else "inverse", help=HELP["forecast_cost"])
        k[2].metric("Billed", money(r.billed), f"{r.billed_pct:.0f}% of committed", delta_color="off", delta_arrow="off", help=HELP["billed"])
        k[3].metric("Verified complete", f"{r.verified_pct:.0f}%", f"{r.verified_pct - r.planned_pct:+.0f} pts vs plan" if pd.notna(r.planned_pct) else None, help=HELP["verified"])
        k = st.columns(4)
        k[0].metric("Schedule index", f"{r.spi:.2f}" if pd.notna(r.spi) else "–", help=HELP["spi"])
        k[1].metric("Forecast finish", fmt_date(r.forecast_finish) if ctx.has_ctl else "–",
                    f"{r.delay_weeks:.0f} wk late" if ctx.has_ctl and pd.notna(r.delay_weeks) and r.delay_weeks > 0 else None,
                    delta_color="inverse", help=HELP["finish"])
        k[2].metric("Penalty exposure", money(r.ld_exposure) if ctx.has_ctl else "–", help=HELP["penalty"])
        k[3].metric("First-pass rate", f"{r.first_pass_pct:.0f}%" if pd.notna(r.first_pass_pct) else "–", help=HELP["first_pass"])
        k = st.columns(4)
        k[0].metric("Flagged: recommend hold", money(r.blocked if pd.notna(r.blocked) else 0), help=HELP["blocked"])
        k[1].metric("To check (REVIEW)", money(r.to_check if pd.notna(r.to_check) else 0), help=HELP["to_check"])
        k[2].metric("Pending changes", money(r.pending_co), help=HELP["pending"])
        cr = ctx.cash.set_index("project_id") if len(ctx.cash) else pd.DataFrame()
        rec = cr.paid_needs_recovery.get(pick, np.nan) if "paid_needs_recovery" in cr else np.nan
        k[3].metric("To recover", money(rec) if pd.notna(rec) else "–", help=HELP["recover"])

    # progress + forecast
    st.markdown("### Progress and forecast")
    pr = ctx.prog[ctx.prog.project_id == pick].copy()
    pr["date"] = pr.period.map(_period_end)
    long = pr.melt(id_vars="date", value_vars=["planned_pct", "verified_pct", "billed_pct"], var_name="s", value_name="pct").dropna()
    long["s"] = long.s.map({"planned_pct": "Planned", "verified_pct": "Verified", "billed_pct": "Billed"})
    last = pr.dropna(subset=["verified_pct"]).tail(1)
    fc = pd.DataFrame()
    if ctx.has_ctl and len(last) and pd.notna(r.forecast_finish):
        fc = pd.DataFrame({"date": [last.date.iloc[0], r.forecast_finish], "pct": [last.verified_pct.iloc[0], 100.0], "s": "Forecast"})
    dom = ["Planned", "Verified", "Billed", "Forecast"]
    rng_ = [SERIES[3], SERIES[2], SERIES[0], SERIES[6]]
    base = alt.Chart(pd.concat([long, fc])).encode(
        x=alt.X("date:T", title=None), y=alt.Y("pct:Q", title="% complete", scale=alt.Scale(domain=[0, 105])),
        color=alt.Color("s:N", scale=alt.Scale(domain=dom, range=rng_), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition("datum.s == 'Planned' || datum.s == 'Forecast'", alt.value([5, 4]), alt.value([1, 0])),
        tooltip=["s", alt.Tooltip("date:T", format="%b %Y"), alt.Tooltip("pct:Q", format=".1f")])
    layers = [base.mark_line(strokeWidth=2), base.mark_point(size=50, filled=True)]
    marks = []
    if ctx.has_ctl and pd.notna(r.planned_finish) and pd.notna(r.forecast_finish) and abs((r.forecast_finish - r.planned_finish).days) < 30:
        marks.append({"d": r.planned_finish, "t": "Contract ≈ forecast finish", "dy": -132})
    else:
        if ctx.has_ctl and pd.notna(r.planned_finish):
            marks.append({"d": r.planned_finish, "t": "Contract finish", "dy": -132})
        if ctx.has_ctl and pd.notna(r.forecast_finish):
            marks.append({"d": r.forecast_finish, "t": "Forecast finish", "dy": -114})
    if marks:
        mk = pd.DataFrame(marks)
        layers.append(alt.Chart(mk).mark_rule(strokeDash=[2, 3], color="#8792a6").encode(x="d:T"))
        for m in marks:
            layers.append(alt.Chart(pd.DataFrame([m])).mark_text(align="left", dx=4, dy=m["dy"], color="#b9c3d3", fontSize=11)
                          .encode(x="d:T", text="t"))
    c1, c2 = st.columns([3, 2])
    c1.altair_chart(alt.layer(*layers).properties(height=320, title="Progress: planned vs built vs billed"), width="stretch")
    gap = (r.verified_pct - r.planned_pct) if pd.notna(r.planned_pct) else np.nan
    ahead = (r.billed_pct - r.verified_pct) if pd.notna(r.billed_pct) else np.nan
    c1.markdown(
        '<div class="howto"><b>How to read:</b> orange dashed = the plan; green = work verified as built; blue = what the '
        'contractor has billed; purple dashed = forecast to completion (earned schedule; ranges in Forecast & risk). '
        + (f'<b>Now:</b> {abs(gap):.0f} pts {"behind" if gap < 0 else "ahead of"} plan (≈ {money(abs(gap) / 100 * r.committed)} of work)'
           if pd.notna(gap) else "")
        + (f'; billed {"ahead of" if ahead > 0.5 else "in line with"} built work'
           + (f' by {money(ahead / 100 * r.committed)}' if ahead > 0.5 else "") if pd.notna(ahead) else "")
        + ".</div>", unsafe_allow_html=True)

    # budget check: is there room in the budget for the changes?
    with c2:
        st.markdown("**Budget check: room in budget vs changes**")
        if ctx.has_ctl and pd.notna(r.budget):
            room = r.budget - r.original_contract
            rows = [{"bar": "Budget room above contract", "part": "Room", "v": room},
                    {"bar": "Changes to contract", "part": "Approved", "v": r.approved_co},
                    {"bar": "Changes to contract", "part": "Pending", "v": r.pending_co}]
            bd = pd.DataFrame(rows)
            bd["vm"] = bd.v / 1e3
            dom = ["Room", "Approved", "Pending"]
            bc = alt.Chart(bd).mark_bar(height=34, stroke="#0e1525", strokeWidth=2).encode(
                y=alt.Y("bar:N", title=None, sort=["Budget room above contract", "Changes to contract"], axis=alt.Axis(labelLimit=220)),
                x=alt.X("sum(vm):Q", title="$ thousands", stack="zero"),
                color=alt.Color("part:N", scale=alt.Scale(domain=dom, range=[SERIES[0], SERIES[1], SERIES[6]]),
                                legend=alt.Legend(orient="top", title=None)),
                order=alt.Order("part:N", sort="ascending"),
                tooltip=["part", alt.Tooltip("v:Q", format="$,.0f", title="Amount")])
            tot = pd.DataFrame([{"bar": "Budget room above contract", "vm": room / 1e3, "t": money(room)},
                                {"bar": "Changes to contract", "vm": (r.approved_co + r.pending_co) / 1e3, "t": money(r.approved_co + r.pending_co)}])
            tl = alt.Chart(tot).mark_text(align="left", dx=6, color="#e8eef8", fontSize=12).encode(
                y=alt.Y("bar:N", sort=["Budget room above contract", "Changes to contract"]), x="vm:Q", text="t")
            st.altair_chart((bc + tl).properties(height=170), width="stretch")
            over = r.eac - r.budget
            st.markdown(
                f'<div class="howto"><span class="kv">{money(r.original_contract)}</span> contract + '
                f'<span class="kv">{money(r.approved_co)}</span> approved + <span class="kv">{money(r.pending_co)}</span> pending = '
                f'<b class="kv">{money(r.eac)}</b> forecast vs <span class="kv">{money(r.budget)}</span> budget → '
                + (f'<b style="color:#ff6b6b">{money(over)} over</b>' if over > 0 else f'<b style="color:#3fd13f">{money(-over)} under</b>')
                + ('. <b>Why:</b> changes exceed the room in the budget'
                   + (f'; deciding the {money(r.pending_co)} of pending changes decides whether this project stays in budget.' if over > 0 and r.pending_co > 0 else '.')
                   if over > 0 else '. The changes fit within the budget room.')
                + '</div>', unsafe_allow_html=True)
        else:
            st.caption("Add project_controls.csv (budget) to see this check.")

    # bills by month
    g = ctx.gf[ctx.gf.project_id == pick]
    c1, c2 = st.columns(2)
    if len(g):
        gg = g.groupby(["period", "gate"]).gross_amount.sum().reset_index()
        gg["gate_l"] = gg.gate.map({"PASS": "✓ Pass", "REVIEW": "! Review", "HOLD": "✕ Hold"})
        bm = alt.Chart(gg).mark_bar(stroke="#0e1525", strokeWidth=2, cornerRadiusEnd=3).encode(
            x=alt.X("period:O", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("gross_amount:Q", title="Billed ($)", axis=alt.Axis(format="$~s")),
            color=alt.Color("gate_l:N", scale=alt.Scale(domain=["✓ Pass", "! Review", "✕ Hold"],
                                                         range=[GATE_COLOR["PASS"], GATE_COLOR["REVIEW"], GATE_COLOR["HOLD"]]),
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=["period", "gate_l", alt.Tooltip("gross_amount:Q", format="$,.0f")]).properties(height=280, title="Bills by month and gate")
        c1.altair_chart(bm, width="stretch")
        held = 100 * g[g.gate == "HOLD"].gross_amount.sum() / g.gross_amount.sum() if g.gross_amount.sum() else 0
        c1.markdown(f'<div class="howto"><b>How to read:</b> each bar is a month of billing; green passed every check, yellow needs a '
                    f'look, red is stopped. <b>Now:</b> {held:.0f}% of billed value is on hold. Mostly-red months point to a contractor '
                    f'billing process problem, not one-off mistakes.</div>', unsafe_allow_html=True)
    e = ctx.ef[ctx.ef.invoice_id.isin(g.invoice_id)]
    e = e[e.amount_at_risk > 0].merge(g[["invoice_id", "prime_id"]], on="invoice_id")
    with c2:
        st.markdown("**Where the money at risk comes from**")
        if len(e):
            has_dup = (e.rule == "DUPLICATE_INVOICE").any()
            excl = st.toggle("Exclude one-off duplicates", value=False, key=f"exdup_{pick}", disabled=not has_dup,
                             help="Duplicates are rare but large. Hide them to see the recurring billing habits.")
            e2 = e[e.rule != "DUPLICATE_INVOICE"] if excl else e
            if len(e2):
                ei = e2.assign(issue=e2.rule.map(lambda x: RULE_LABEL.get(x, x)),
                               contractor=e2.prime_id.map(lambda p: ctx.names.get(p, p)),
                               bills=1).groupby(["issue", "contractor"]).agg(v=("amount_at_risk", "sum"), bills=("bills", "sum")).reset_index()
                totals = ei.groupby("issue").agg(v=("v", "sum"), bills=("bills", "sum")).reset_index()
                totals["t"] = [f"{money(v)} · {b} line{'s' if b != 1 else ''}" for v, b in zip(totals.v, totals.bills)]
                order = list(totals.sort_values("v", ascending=False).issue)
                cdom = sorted(ei.contractor.unique())
                ib = alt.Chart(ei).mark_bar(height=16, stroke="#0e1525", strokeWidth=1).encode(
                    y=alt.Y("issue:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
                    x=alt.X("sum(v):Q", title="Money at risk ($)", axis=alt.Axis(format="$~s"), scale=alt.Scale(nice=True)),
                    color=alt.Color("contractor:N", scale=alt.Scale(domain=cdom, range=SERIES[:len(cdom)]),
                                    legend=alt.Legend(orient="bottom", title=None, columns=1, labelLimit=320)),
                    tooltip=["issue", "contractor", alt.Tooltip("v:Q", format="$,.0f"), "bills"])
                tx = alt.Chart(totals).mark_text(align="left", dx=6, color="#e8eef8", fontSize=11).encode(
                    y=alt.Y("issue:N", sort=order), x="v:Q", text="t")
                st.altair_chart((ib + tx).properties(height=alt.Step(30)), width="stretch")
                top = totals.sort_values("v", ascending=False).iloc[0]
                share = 100 * top.v / totals.v.sum()
                lead = ei.groupby("contractor").v.sum().sort_values(ascending=False)
                st.markdown(f'<div class="howto"><b>How to read:</b> bar length = money on stopped or flagged lines; color = contractor. '
                            f'<b>Now:</b> "{top.issue}" is {share:.0f}% of it; {lead.index[0]} accounts for {100 * lead.iloc[0] / lead.sum():.0f}%. '
                            'Big single bars are one-off events (block them with a rule); issues repeated over many lines are habits '
                            '(fix them with the contractor).</div>', unsafe_allow_html=True)
            else:
                st.success("Nothing left once one-off duplicates are excluded.")
        else:
            st.success("No money at risk on this project's bills.")

    # rates vs program median
    st.markdown("### Rates vs program median")
    po = ctx.tables["purchase_orders"]
    pol = ctx.tables["po_lines"].merge(po[["po_id", "project_id"]], on="po_id")
    pol = pol[(pol.line_source == "CONTRACT") & (pol.unit.str.upper() != "LS") & pol.project_id.isin(ctx.pt.project_id)]
    med = pol.groupby(["item_code", "unit"]).contract_unit_rate.median().rename("median")
    shared = pol.groupby(["item_code", "unit"]).project_id.nunique().rename("n_projects")
    mine = pol[pol.project_id == pick].merge(med, on=["item_code", "unit"]).merge(shared, on=["item_code", "unit"])
    mine = mine[mine.n_projects >= 2]
    if len(mine):
        mine = mine.assign(vs_median_pct=100 * (mine.contract_unit_rate / mine["median"] - 1),
                           value=mine.contract_unit_rate * mine.contract_qty)
        top = mine.sort_values("value", ascending=False).head(12)
        rb = alt.Chart(top).mark_bar(cornerRadiusEnd=3, height=14).encode(
            y=alt.Y("description:N", sort=alt.EncodingSortField("value", order="descending"), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("vs_median_pct:Q", title="% vs program median rate"),
            color=alt.condition("datum.vs_median_pct > 0", alt.value(SERIES[1]), alt.value(SERIES[0])),
            tooltip=["item_code", "description", "unit", alt.Tooltip("contract_unit_rate:Q", format="$,.2f"),
                     alt.Tooltip("median:Q", format="$,.2f"), alt.Tooltip("vs_median_pct:Q", format="+.1f")]
        ).properties(height=alt.Step(24), title="Largest items by value (orange = above median)")
        st.altair_chart(rb + alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8792a6").encode(x="x:Q"), width="stretch")
    else:
        st.caption("No items shared with other projects to compare.")

    # project-level live insights + evidence
    st.markdown("### Live insights")
    name = r["name"]
    mine_ins = [d for d in ctx.insights if name in d["title"] or name in d["detail"] or pick in d["detail"]]
    if not mine_ins and not h["alerts"]:
        st.success("Nothing needs a decision on this project right now.")
    for d in mine_ins[:6]:
        col = {"critical": "#ff6b6b", "warning": "#fab219", "info": "#7fb8ff", "good": "#3fd13f"}[d["level"]]
        st.markdown(f'<div style="border-left:3px solid {col};padding:6px 12px;margin:6px 0;background:rgba(14,21,37,.5);border-radius:6px">'
                    f'<b>{d["title"]}</b><br>{d["detail"]}<br><span class="muted">Action: {d["action"]}</span></div>', unsafe_allow_html=True)
    with st.expander(f"Evidence: {len(g)} bills on this project"):
        view = g.merge(ctx.risk, on="invoice_id", how="left").sort_values("period")
        st.dataframe(view[["invoice_id", "period", "gate", "gross_amount", "blocked", "to_check", "rules_triggered"]],
                     width="stretch", hide_index=True, column_config={
                         "gross_amount": st.column_config.NumberColumn("Billed", format="$%,.0f"),
                         "blocked": st.column_config.NumberColumn("Blocked", format="$%,.0f"),
                         "to_check": st.column_config.NumberColumn("To check", format="$%,.0f")})
    st.caption(f"Data as of {fmt_date(r.data_date) if ctx.has_ctl and pd.notna(r.data_date) else ctx.end_period}. "
               "Figures recalculate whenever the source data changes.")
