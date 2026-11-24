"""
F24: the five-chapter leadership story and its exports (PDF and PowerPoint).

build_story(...) turns the program results into chapters: headline, three supporting points and the data for one chart.
The app draws the charts with Altair; the exports redraw them with matplotlib so the files stand alone.
"""
import io
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from insights import RULE_LABEL, money  # noqa: E402

INK, MUTED, BLUE, ORANGE, GREEN, AMBER, RED = "#1b2432", "#5f6b7e", "#3987e5", "#d95926", "#199e70", "#c98500", "#d03b3b"


def build_story(program, pt, prog, gf, ef, risk, rc, perf, insights, health, labels, end_period):
    ch = []
    committed = pt.committed.sum()
    ver = 100 * pt.verified.sum() / committed if committed else np.nan
    plan = (pt.planned_pct * pt.committed).sum() / committed if committed and pt.planned_pct.notna().any() else np.nan
    late = pt[pt.delay_weeks > 0] if "delay_weeks" in pt else pt.iloc[0:0]
    hl = (health.health.fillna(0) * pt.set_index("project_id").committed.reindex(health.project_id).values).sum() / committed if committed else np.nan
    agg = prog[prog.project_id.isin(pt.project_id)].groupby("period").apply(lambda d: pd.Series({
        "Planned": (d.planned_pct * d.committed).sum() / d.committed.sum() if d.planned_pct.notna().any() else np.nan,
        "Verified": 100 * d.verified.sum() / d.committed.sum(), "Billed": 100 * d.billed.sum() / d.committed.sum()}), include_groups=False).reset_index()
    agg.loc[agg.period > end_period, ["Verified", "Billed"]] = np.nan  # no actuals after the data month
    ch.append({"title": "1. Where we stand",
               "headline": f"{program} is {ver:.0f}% built against a plan of {plan:.0f}%" if pd.notna(plan) else f"{program} is {ver:.0f}% built",
               "points": [f"{money(committed)} committed across {len(pt)} projects; program health {hl:.0f}/100.",
                          f"{len(late)} of {len(pt)} projects forecast to finish late" + (f", worst {labels.get(late.sort_values('delay_weeks').project_id.iloc[-1], '')} "
                                                                                          f"({late.delay_weeks.max():.0f} weeks)." if len(late) else "."),
                          f"Billed {100 * pt.billed.sum() / committed:.0f}% vs verified {ver:.0f}%: " +
                          ("paid ahead of work in place." if pt.billed.sum() > pt.verified.sum() * 1.01 else "billing is in line with work in place.")],
               "chart": ("lines", agg)})
    blocked, check = risk.blocked.sum(), risk.to_check.sum()
    rr = rc.head(6) if len(rc) else pd.DataFrame(columns=["root_cause", "money_at_risk"])
    ch.append({"title": "2. Where money is at risk",
               "headline": f"{money(blocked)} is blocked and {money(check)} needs checking",
               "points": [f"{(gf.gate == 'HOLD').sum()} of {len(gf)} bills are on hold; first-pass rate {100 * (gf.gate == 'PASS').mean():.0f}%.",
                          (f"Largest cause: {rr.root_cause.iloc[0].lower()} ({money(rr.money_at_risk.iloc[0])})." if len(rr) else "No failed checks."),
                          f"At risk = {100 * (blocked + check) / max(gf.gross_amount.sum(), 1):.1f}% of billing (industry studies put duplicate and "
                          "overpayment leakage at roughly 1-3%)."],
               "chart": ("barh", rr.rename(columns={"root_cause": "label", "money_at_risk": "value"})[["label", "value"]])})
    pf = perf.sort_values("score") if len(perf) else perf
    ch.append({"title": "3. Why it happens",
               "headline": (f"{pf.contractor.iloc[0]} scores {pf.score.iloc[0]:.0f}/100: the weakest billing process" if len(pf) else "Contractor performance"),
               "points": [f"{r.contractor}: score {r.score:.0f}, first-pass {r.first_pass_pct:.0f}%, {r.action.lower()}." for r in pf.head(3).itertuples()]
               or ["No contractor data."],
               "chart": ("barh", pf.rename(columns={"contractor": "label", "score": "value"})[["label", "value"]] if len(pf) else pd.DataFrame(columns=["label", "value"]))})
    bud = pt[["name", "budget", "eac"]].copy() if "budget" in pt else pt[["name", "eac"]].assign(budget=np.nan)
    over = pt[pt.vac < 0] if "vac" in pt else pt.iloc[0:0]
    ch.append({"title": "4. What it will cost",
               "headline": (f"Forecast {money(pt.eac.sum())} vs budget {money(pt.budget.sum())}" if pt.budget.notna().any() else f"Forecast {money(pt.eac.sum())}"),
               "points": [f"{len(over)} project(s) forecast over budget" + (f", largest {over.sort_values('vac').name.iloc[0]} ({money(-over.vac.min())})." if len(over) else "."),
                          f"Pending change orders add {money(pt.pending_co.sum())} if approved.",
                          f"Delay penalty exposure {money(pt.ld_exposure.sum())} if forecast delays happen." if "ld_exposure" in pt else ""],
               "chart": ("pair", bud)})
    top = insights[:5]
    ch.append({"title": "5. Decisions this week",
               "headline": f"{len(top)} decisions, ranked by urgency and dollars",
               "points": [f"{d['title']}: {d['action']}" for d in top] or ["Nothing needs a decision."],
               "chart": None})
    return ch


def _chart_png(chart, w=7.5, h=3.2):
    if chart is None:
        return None
    kind, df = chart
    if df is None or not len(df):
        return None
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    if kind == "lines":
        for col, c, ls in [("Planned", AMBER, "--"), ("Verified", GREEN, "-"), ("Billed", BLUE, "-")]:
            if col in df:
                ax.plot(df.period, df[col], color=c, ls=ls, lw=2, marker="o", ms=3, label=col)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% complete", color=MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    elif kind == "barh":
        d = df.sort_values("value")
        big = d.value.abs().max() > 1e4
        ax.barh(d.label.astype(str).str.slice(0, 64), d.value / (1e6 if big else 1), color=ORANGE, height=0.6)
        if big:
            ax.set_xlabel("$ millions", color=MUTED, fontsize=8)
        ax.tick_params(axis="y", labelsize=7.5)
    elif kind == "pair":
        y = np.arange(len(df))
        ax.barh(y - 0.2, df.budget / 1e6, 0.38, color=BLUE, label="Budget")
        ax.barh(y + 0.2, df.eac / 1e6, 0.38, color=ORANGE, label="Forecast")
        ax.set_yticks(y, df.name)
        ax.set_xlabel("$ millions", color=MUTED, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def story_pdf(chapters, program, author="Manish Kumar"):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    ss = getSampleStyleSheet()
    h = ParagraphStyle("h", parent=ss["Title"], alignment=0, textColor=HexColor(INK), fontSize=22)
    sub = ParagraphStyle("s", parent=ss["Normal"], textColor=HexColor(MUTED), fontSize=10)
    hl = ParagraphStyle("hl", parent=ss["Heading2"], textColor=HexColor(INK), fontSize=15)
    pt = ParagraphStyle("p", parent=ss["Normal"], fontSize=10.5, leading=14, leftIndent=10)
    story = [Paragraph(f"{program}: executive brief", h),
             Paragraph(f"Multi-Prime Commercial Control Tower · {date.today():%d %b %Y} · prepared by {author}", sub), Spacer(1, 14)]
    for i, c in enumerate(chapters):
        if i:
            story.append(PageBreak())
        story += [Paragraph(c["title"], sub), Paragraph(c["headline"], hl)]
        story += [Paragraph("• " + p, pt) for p in c["points"] if p]
        png = _chart_png(c["chart"])
        if png:
            story += [Spacer(1, 8), Image(io.BytesIO(png), width=8.6 * inch, height=3.67 * inch)]
    story += [Spacer(1, 12), Paragraph("Demonstration data unless stated otherwise. Rules decide payment; models prioritize and explain.", sub)]
    doc.build(story)
    return buf.getvalue()


def story_pptx(chapters, program, author="Manish Kumar"):
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    def text(slide, x, y, w, h, s, size, bold=False, color=INK):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = s
        p.font.size, p.font.bold = Pt(size), bold
        p.font.color.rgb = RGBColor.from_string(color.lstrip("#"))
        return tf

    s = prs.slides.add_slide(blank)
    text(s, 0.8, 2.4, 11.5, 1.2, f"{program}", 40, True)
    text(s, 0.8, 3.6, 11.5, 0.8, "Executive brief · Multi-Prime Commercial Control Tower", 20, color=MUTED)
    text(s, 0.8, 4.3, 11.5, 0.6, f"{date.today():%d %B %Y} · {author}", 14, color=MUTED)
    for c in chapters:
        s = prs.slides.add_slide(blank)
        text(s, 0.6, 0.35, 12, 0.4, c["title"], 12, color=MUTED)
        text(s, 0.6, 0.7, 12, 0.9, c["headline"], 26, True)
        png = _chart_png(c["chart"], 7.2, 4.2)
        bx = 0.6 if png else 0.6
        tf = text(s, bx, 1.8, 5.0 if png else 12, 4.8, "", 14)
        first = True
        for ptxt in [p for p in c["points"] if p]:
            para = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            para.text = "• " + ptxt
            para.font.size = Pt(14)
            para.space_after = Pt(10)
        if png:
            s.shapes.add_picture(io.BytesIO(png), Inches(5.8), Inches(1.8), width=Inches(7.0))
        text(s, 0.6, 7.0, 12, 0.3, "Demonstration data unless stated otherwise.", 9, color=MUTED)
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
