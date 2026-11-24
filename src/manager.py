"""
Program-manager layer: turns invoice checks into budget, schedule, cash, root-cause,
contractor and policy information.

Optional input tables (the dashboard works without them, with fewer views):
  project_controls.csv  one row per project: budget, planned/forecast finish, delay-penalty terms,
                        advance paid and how it is recovered, payment terms
  baseline_plan.csv     planned cumulative % complete by month (schedule baseline)
  payments.csv          what was paid on each invoice, when, advance recovered, amount withheld
"""
from pathlib import Path

import numpy as np
import pandas as pd

from insights import RULE_LABEL, money, prime_names

CONTROL_SCHEMA = {
    "project_controls": [
        ("project_id", "str", "Project id (matches invoices.project_id)"),
        ("short_name", "str", "Readable name shown on screens, e.g. Amba"),
        ("approved_budget", "float", "Owner's approved budget for this contract, including contingency ($)"),
        ("planned_start", "date", "Baseline start date YYYY-MM-DD"),
        ("planned_finish", "date", "Baseline (contract) completion date"),
        ("forecast_finish", "date", "Current forecast completion (leave blank to let the dashboard estimate it)"),
        ("data_date", "date", "Date the data is current to"),
        ("ld_rate_pct_per_week", "float", "Delay penalty (liquidated damages) per week of delay, % of contract value"),
        ("ld_cap_pct", "float", "Maximum delay penalty, % of contract value"),
        ("advance_paid", "float", "Mobilization / advance payment made to the contractor ($, 0 if none)"),
        ("advance_recovery_start_pct", "float", "Recovery of the advance starts at this % of progress"),
        ("advance_recovery_end_pct", "float", "Advance must be fully recovered by this % of progress"),
        ("payment_terms_days", "float", "Days from invoice to payment allowed by the contract"),
    ],
    "baseline_plan": [
        ("project_id", "str", "Project id"),
        ("period", "str", "YYYY-MM"),
        ("planned_pct_cum", "float", "Planned cumulative % complete at the end of this month (0-100)"),
    ],
    "payments": [
        ("invoice_id", "str", "Invoice id"),
        ("submitted_date", "date", "Date the invoice was received"),
        ("certified_date", "date", "Date it was certified (optional)"),
        ("paid_date", "date", "Date paid (blank if not paid yet)"),
        ("amount_paid", "float", "Amount paid ($)"),
        ("advance_recovered", "float", "Advance recovered from this invoice ($)"),
        ("amount_withheld", "float", "Amount withheld from this invoice for disputed items ($)"),
        ("status", "str", "Free text, e.g. Paid, Under scrutiny"),
    ],
}

# What each failed check usually means upstream, who owns the fix, and the process fix
ROOT_CAUSE = {
    "DUPLICATE_INVOICE": ("Pay-app numbering not controlled", "Accounts payable",
                          "Make the pay-app number unique per contract per month in the intake system"),
    "QTY_OVER_RECEIPT": ("Bills prepared before field measurement is agreed", "Field / quantity surveyor",
                         "Agree quantities jointly before the bill is prepared; bill from the signed record only"),
    "PCT_COMPLETE_OVERBILL": ("Progress on lump-sum items claimed, not measured", "Project controls",
                              "Define measurable milestones for each lump-sum item"),
    "RATE_OVER_CONTRACT": ("Contract rates not loaded into the contractor's billing", "Procurement / contracts",
                           "Issue a rate sheet at contract award and lock billing to it"),
    "UNAPPROVED_CO": ("Change control bypassed (work done before approval)", "Program manager",
                      "Enforce no-approval-no-work; clear pending change orders weekly"),
    "TM_OVER_APPROVED": ("Extra-work hours not capped in the field", "Site manager",
                         "Daily sign-off of T&M sheets against the approved hours"),
    "TM_NOT_PREAPPROVED": ("Extra work started without approval", "Site manager",
                           "Pre-approval required before any T&M work"),
    "PER_DIEM_OVER_POLICY": ("Allowance policy not applied", "Contractor's payroll",
                             "Restate the cap in the contract and check headcount against site logs"),
    "RETAINAGE_ERROR": ("Billing template applies the wrong retention", "Contractor's billing",
                        "Fix the template once; spot-check the first bill of every contract"),
    "EXTENSION_ERROR": ("Manual spreadsheet bills", "Contractor's billing",
                        "Require system-generated bills"),
    "HEADER_MISMATCH": ("Manual spreadsheet bills", "Contractor's billing", "Require system-generated bills"),
    "NO_RECEIPT": ("Work billed before it is inspected", "Field / quantity surveyor",
                   "Inspect and record before the billing cut-off"),
}


# ---------------------------------------------------------------- loading
def read_controls(folder):
    folder = Path(folder)
    out = {}
    for t in CONTROL_SCHEMA:
        f = folder / f"{t}.csv"
        if f.exists():
            out[t] = pd.read_csv(f, dtype={"project_id": str, "invoice_id": str, "period": str})
    return out


def coerce_controls(tables):
    out = {}
    for t, cols in CONTROL_SCHEMA.items():
        if t not in tables:
            continue
        df = tables[t].copy()
        for c, typ, _ in cols:
            if c not in df.columns:
                df[c] = np.nan
            if typ == "float":
                df[c] = pd.to_numeric(df[c], errors="coerce")
            elif typ == "date":
                df[c] = pd.to_datetime(df[c], errors="coerce")
            else:
                df[c] = df[c].astype("object").where(df[c].notna(), None)
        out[t] = df
    return out


def project_labels(tables, controls=None):
    lab = {}
    pr = tables.get("projects")
    if pr is not None:
        for col in ("short_name", "campus"):
            if col in pr.columns:
                lab.update({p: n for p, n in zip(pr.project_id, pr[col]) if isinstance(n, str) and n and p not in lab})
    pc = (controls or {}).get("project_controls")
    if pc is not None:
        lab.update({p: n for p, n in zip(pc.project_id, pc.short_name) if isinstance(n, str) and n})
    return lab


def _scope_po(tables, projects, primes):
    po = tables["purchase_orders"][["po_id", "project_id", "prime_id", "retainage_rate"]]
    if projects:
        po = po[po.project_id.isin(projects)]
    if primes:
        po = po[po.prime_id.isin(primes)]
    return po


def duplicate_ids(gates):
    return set(gates.loc[gates.rules_triggered.fillna("").str.contains("DUPLICATE_INVOICE"), "invoice_id"])


def split_risk(gates, exceptions):
    """Split each invoice's money at risk into blocked (HOLD checks) and to-check (REVIEW checks)."""
    e = exceptions[exceptions.amount_at_risk > 0]
    by = e.pivot_table(index="invoice_id", columns="severity", values="amount_at_risk", aggfunc="sum", fill_value=0)
    g = gates[["invoice_id", "amount_at_risk"]].set_index("invoice_id")
    hold = by.get("HOLD", pd.Series(dtype=float)).reindex(g.index, fill_value=0)
    rev = by.get("REVIEW", pd.Series(dtype=float)).reindex(g.index, fill_value=0)
    tot = (hold + rev).replace(0, np.nan)
    scale = (g.amount_at_risk / tot).fillna(0)
    return pd.DataFrame({"blocked": (hold * scale).round(2), "to_check": (rev * scale).round(2)}).reset_index()


# ---------------------------------------------------------------- progress (planned / verified / billed)
def progress_by_period(tables, gates, controls=None, projects=None, primes=None):
    """Cumulative verified and billed value and % by project and month, plus the planned %."""
    po = _scope_po(tables, projects, primes)
    pol = tables["po_lines"].merge(po[["po_id", "project_id"]], on="po_id")
    committed = pol.groupby("project_id").scheduled_value.sum()

    gr = tables["goods_receipts"].merge(po[["po_id", "project_id"]], on="po_id").merge(
        tables["po_lines"][["po_id", "item_code", "contract_unit_rate", "scheduled_value"]], on=["po_id", "item_code"], how="left")
    dup = duplicate_ids(gates)
    inv = tables["invoices"][~tables["invoices"].invoice_id.isin(dup)][["invoice_id", "po_id", "period", "gross_amount"]]
    inv = inv.merge(po[["po_id", "project_id"]], on="po_id")
    lines = tables["invoice_lines"].merge(inv[["invoice_id", "project_id", "period"]], on="invoice_id")
    lines = lines[lines.line_type.isin(["SOV", "CO"])]

    periods = sorted(set(gr.period.dropna()) | set(inv.period.dropna()))
    base = (controls or {}).get("baseline_plan")
    if base is not None:
        periods = sorted(set(periods) | set(base.period.dropna()))
    rows = []
    for p in committed.index:
        g = gr[gr.project_id == p]
        u = g[g.verified_qty.notna()]
        ls = g[g.verified_pct_complete_cum.notna()]
        b = lines[lines.project_id == p]
        last_data = max(set(g.period.dropna()) | set(inv[inv.project_id == p].period.dropna()) or {""})
        for per in periods:
            v_unit = (u[u.period <= per].verified_qty * u[u.period <= per].contract_unit_rate).sum()
            l2 = ls[ls.period <= per].sort_values("period").groupby("item_code").agg(
                pct=("verified_pct_complete_cum", "last"), sv=("scheduled_value", "first"))
            v_ls = (l2.pct * l2.sv / 100).sum()
            rows.append({"project_id": p, "period": per, "committed": committed[p],
                         "verified": v_unit + v_ls if per <= last_data else np.nan,
                         "billed": b[b.period <= per].amount.sum() if per <= last_data else np.nan})
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df["verified_pct"] = 100 * df.verified / df.committed
    df["billed_pct"] = 100 * df.billed / df.committed
    if base is not None:
        df = df.merge(base[["project_id", "period", "planned_pct_cum"]], on=["project_id", "period"], how="left")
        df["planned_pct"] = df.groupby("project_id").planned_pct_cum.transform(lambda s: s.ffill())
        df = df.drop(columns="planned_pct_cum")
    else:
        df["planned_pct"] = np.nan
    return df


def _months_between(a, b):
    return (pd.Period(b, "M") - pd.Period(a, "M")).n


def _pace_finish(prog_p, end_period, data_date):
    """Fallback: completion from the recent pace of verified progress (last 3 months with progress)."""
    s = prog_p[(prog_p.period <= end_period) & prog_p.verified_pct.notna()].sort_values("period")
    if len(s) < 2:
        return pd.NaT
    s = s.iloc[-4:]
    gain = s.verified_pct.iloc[-1] - s.verified_pct.iloc[0]
    months = max(1, _months_between(s.period.iloc[0], s.period.iloc[-1]))
    pace = gain / months
    remaining = 100 - s.verified_pct.iloc[-1]
    if remaining <= 0.5:
        return data_date
    if pace <= 0.1:
        return pd.NaT
    return data_date + pd.Timedelta(days=round(30.4 * remaining / pace))


def _earned_schedule_finish(prog_p, end_period, data_date, planned_finish):
    """Earned schedule: the date at which the plan reached today's verified %, compared with today, gives the
    schedule slip in time; the forecast finish is the planned finish plus that slip. Backtests in the app showed
    plan-based forecasts beat the recent-pace method by a wide margin."""
    s = prog_p.dropna(subset=["planned_pct"]).sort_values("period")
    v = prog_p[(prog_p.period <= end_period) & prog_p.verified_pct.notna()].sort_values("period")
    if not len(s) or not len(v) or pd.isna(planned_finish):
        return pd.NaT
    ev = float(v.verified_pct.iloc[-1])
    t = s.period.map(lambda q: pd.Period(q, "M").end_time.normalize())
    pv = s.planned_pct.values
    if ev >= pv.max() - 1e-9:
        es = t.iloc[int(np.argmax(pv >= pv.max() - 1e-9))]
    else:
        i = int(np.argmax(pv > ev))
        if i == 0:
            es = t.iloc[0] - pd.Timedelta(days=30.4 * (pv[0] - ev) / max(pv[0], 1e-6))
        else:
            frac = (ev - pv[i - 1]) / max(pv[i] - pv[i - 1], 1e-6)
            es = t.iloc[i - 1] + (t.iloc[i] - t.iloc[i - 1]) * frac
    slip = data_date - es
    return planned_finish + slip


def _forecast_finish(prog_p, end_period, data_date, planned_finish=pd.NaT):
    f = _earned_schedule_finish(prog_p, end_period, data_date, planned_finish)
    return f if pd.notna(f) else _pace_finish(prog_p, end_period, data_date)


# ---------------------------------------------------------------- program table (one row per project)
def program_table(tables, gates, exceptions, controls, projects, primes, end_period, labels=None):
    """Budget, forecast, progress, schedule, delay penalty and billing quality per project, to end_period."""
    po = _scope_po(tables, projects, primes)
    pol = tables["po_lines"].merge(po[["po_id", "project_id"]], on="po_id")
    contract = pol[pol.line_source == "CONTRACT"].groupby("project_id").scheduled_value.sum().rename("original_contract")
    committed = pol.groupby("project_id").scheduled_value.sum().rename("committed")
    co = tables["change_orders"].merge(po[["po_id", "project_id"]], on="po_id", suffixes=("", "_po"))
    pending = co[co.status == "PENDING"].groupby("project_id").co_amount.sum().rename("pending_co")
    approved_co = co[co.status == "APPROVED"].groupby("project_id").co_amount.sum().rename("approved_co")
    df = pd.concat([contract, approved_co, committed, pending], axis=1).fillna(0)
    df.index.name = "project_id"
    df["name"] = [(labels or {}).get(p, p) for p in df.index]
    names = prime_names(tables)
    df["contractor"] = po.groupby("project_id").prime_id.apply(
        lambda s: names.get(s.iloc[0], s.iloc[0]) if s.nunique() == 1 else f"{s.nunique()} prime contractors")

    prog = progress_by_period(tables, gates, controls, projects, primes)
    at = prog[prog.period <= end_period].sort_values("period")
    last = at.dropna(subset=["verified"]).groupby("project_id").last()
    plan_now = at.groupby("project_id").planned_pct.last()
    df["billed"] = last.billed
    df["verified"] = last.verified
    df["billed_pct"] = last.billed_pct
    df["verified_pct"] = last.verified_pct
    df["planned_pct"] = plan_now
    df["spi"] = df.verified_pct / df.planned_pct.where(df.planned_pct > 0)
    df["billed_ahead"] = df.billed - df.verified

    # forecast cost: committed + pending change orders (assume they are approved: the cautious view)
    df["eac"] = df.committed + df.pending_co
    pc = (controls or {}).get("project_controls")
    if pc is not None:
        pc = pc.set_index("project_id")
        df["budget"] = pc.approved_budget
        df["vac"] = df.budget - df.eac
        df["planned_finish"] = pc.planned_finish
        df["data_date"] = pc.data_date
        df["ld_rate"] = pc.ld_rate_pct_per_week
        df["ld_cap"] = pc.ld_cap_pct
        given = pc.forecast_finish
        est = {}
        for p in df.index:
            dd = df.at[p, "data_date"] if pd.notna(df.at[p, "data_date"]) else pd.Period(end_period, "M").end_time.normalize()
            est[p] = _forecast_finish(prog[prog.project_id == p], end_period, dd, df.at[p, "planned_finish"])
        df["forecast_finish"] = given.reindex(df.index).where(given.reindex(df.index).notna(), pd.Series(est))
        df["forecast_source"] = np.where(given.reindex(df.index).notna(), "Given", "Earned schedule")
        delay = (df.forecast_finish - df.planned_finish).dt.days / 7
        df["delay_weeks"] = delay.clip(lower=0).round(1)
        df["ld_exposure"] = (np.minimum(df.ld_rate * df.delay_weeks, df.ld_cap) / 100 * df.committed).fillna(0).round(0)
    else:
        for c in ("budget", "vac", "planned_finish", "forecast_finish", "delay_weeks", "ld_exposure", "data_date"):
            df[c] = np.nan

    # billing quality in the selected window is added by the caller; status here uses cost + schedule
    reasons, status = [], []
    for p, r in df.iterrows():
        why, lvl = [], 0
        if pd.notna(r.vac) and r.vac < 0:
            why.append(f"forecast {money(-r.vac)} over budget")
            lvl = 2
        elif pd.notna(r.vac) and pd.notna(r.budget) and r.vac < 0.02 * r.budget:
            why.append("under 2% budget headroom")
            lvl = max(lvl, 1)
        if pd.notna(r.delay_weeks) and r.delay_weeks > 4:
            why.append(f"{r.delay_weeks:.0f} weeks late")
            lvl = 2
        elif pd.notna(r.delay_weeks) and r.delay_weeks > 0:
            why.append(f"{r.delay_weeks:.0f} week{'s' if round(r.delay_weeks) != 1 else ''} late")
            lvl = max(lvl, 1)
        if pd.notna(r.spi) and r.spi < 0.9:
            why.append(f"behind plan (schedule index {r.spi:.2f})")
            lvl = max(lvl, 1)
        if pd.notna(r.billed_ahead) and r.committed and r.billed_ahead > 0.02 * r.committed:
            why.append(f"billed {money(r.billed_ahead)} ahead of verified work")
            lvl = max(lvl, 1)
        reasons.append("; ".join(why) or "on budget and on time")
        status.append(["Green", "Amber", "Red"][lvl])
    df["status"], df["why"] = status, reasons
    return df.reset_index(), prog


# ---------------------------------------------------------------- root causes
def root_causes(gates_f, exc_f):
    """Pareto of failed checks with the upstream cause, the owner and the first-pass rate if fixed."""
    if not len(gates_f):
        return pd.DataFrame()
    ids = set(gates_f.invoice_id)
    e = exc_f[exc_f.invoice_id.isin(ids)]
    if not len(e):
        return pd.DataFrame()
    agg = e.groupby("rule").agg(invoices=("invoice_id", "nunique"), lines=("invoice_id", "size"),
                                at_risk=("amount_at_risk", lambda s: s.clip(lower=0).sum())).reset_index()
    agg = agg.sort_values(["invoices", "at_risk"], ascending=False)
    rules_by_inv = e.groupby("invoice_id").rule.apply(set)
    n = len(gates_f)
    fixed, out = set(), []
    for r in agg.itertuples():
        fixed.add(r.rule)
        still_failing = sum(1 for s in rules_by_inv if not s <= fixed)
        rc, owner, fix = ROOT_CAUSE.get(r.rule, ("Other", "Program manager", "Investigate"))
        out.append({"check_failed": RULE_LABEL.get(r.rule, r.rule), "rule": r.rule, "root_cause": rc, "owner": owner,
                    "process_fix": fix, "invoices_affected": r.invoices, "pct_of_failed_invoices": 0.0,
                    "money_at_risk": round(r.at_risk, 0),
                    "first_pass_if_fixed": round(100 * (n - still_failing) / n, 0)})
    df = pd.DataFrame(out)
    failed = len(rules_by_inv)
    df["pct_of_failed_invoices"] = (100 * df.invoices_affected / failed).round(0)
    df.attrs["first_pass_now"] = 100 * (n - failed) / n
    return df


# ---------------------------------------------------------------- contractor performance
def contractor_performance(gates_f, exc_f, prog_table):
    if not len(gates_f):
        return pd.DataFrame()
    risk = split_risk(gates_f, exc_f)
    g = gates_f.merge(risk, on="invoice_id", how="left")
    rows = []
    per_all = sorted(g.period.unique())
    for prm, grp in g.groupby("prime_id"):
        fp = 100 * (grp.gate == "PASS").mean()
        by_p = grp.groupby("period").gate.apply(lambda s: 100 * (s == "PASS").mean())
        trend = "Not enough months"
        if len(by_p) >= 3:
            early, late = by_p.iloc[:-2].mean(), by_p.iloc[-2:].mean()
            # small samples are noisy: only call a trend on a clear move (20+ points)
            trend = "Improving" if late - early >= 20 else "Worsening" if early - late >= 20 else "Steady"
        e = exc_f[exc_f.invoice_id.isin(grp.invoice_id)].merge(grp[["invoice_id", "period"]], on="invoice_id")
        repeat = e.groupby("rule").period.nunique()
        repeat = [RULE_LABEL.get(r, r) for r in repeat[repeat >= 2].sort_values(ascending=False).index[:3]]
        billed = grp.gross_amount.sum()
        risk_pct = 100 * grp.amount_at_risk.sum() / billed if billed else 0
        rows.append({"prime_id": prm, "invoices": len(grp), "billed": billed, "first_pass_pct": round(fp, 0),
                     "trend": trend, "risk_pct_of_billed": round(risk_pct, 1),
                     "blocked": grp.blocked.sum(), "to_check": grp.to_check.sum(),
                     "repeat_issues": "; ".join(repeat) or "-"})
    return pd.DataFrame(rows)


def score_contractors(perf, names, prog_table):
    if not len(perf):
        return perf
    perf = perf.copy()
    perf["contractor"] = perf.prime_id.map(lambda p: names.get(p, p))
    spi = prog_table.groupby("contractor").spi.mean() if "spi" in prog_table else pd.Series(dtype=float)
    perf["schedule_spi"] = perf.contractor.map(spi).round(2)
    s_fp = perf.first_pass_pct.clip(0, 100)
    s_risk = (100 - 10 * perf.risk_pct_of_billed).clip(0, 100)
    s_spi = (100 * perf.schedule_spi.clip(upper=1)).fillna(s_fp * 0 + 75)
    perf["score"] = (0.4 * s_fp + 0.3 * s_risk + 0.3 * s_spi).round(0)
    act = []
    for r in perf.itertuples():
        if r.score < 60 or (r.trend == "Worsening" and r.score < 80):
            act.append("Performance meeting")
        elif r.score < 75:
            act.append("Watch")
        else:
            act.append("Reward: lighter review")
    perf["action"] = act
    return perf.sort_values("score")


# ---------------------------------------------------------------- rate benchmarking (across contractors)
def rate_benchmark(tables, projects=None, primes=None, labels=None):
    labels = labels or {}
    po = _scope_po(tables, projects, primes)
    pol = tables["po_lines"].merge(po[["po_id", "project_id", "prime_id"]], on="po_id")
    pol = pol[(pol.line_source == "CONTRACT") & (pol.unit.str.upper() != "LS") & (pol.contract_qty > 0)]
    shared = pol.groupby(["item_code", "unit"]).project_id.nunique()
    shared = shared[shared >= 2].index
    pol = pol.set_index(["item_code", "unit"]).loc[lambda d: d.index.isin(shared)].reset_index()
    if not len(pol):
        return pd.DataFrame(), pd.DataFrame()
    med = pol.groupby(["item_code", "unit"]).contract_unit_rate.median().rename("median_rate")
    pol = pol.merge(med, on=["item_code", "unit"])
    pol["excess_vs_median"] = ((pol.contract_unit_rate - pol.median_rate).clip(lower=0) * pol.contract_qty)
    items = pol.groupby(["item_code", "unit"]).agg(
        description=("description", "first"), projects=("project_id", "nunique"),
        lowest=("contract_unit_rate", "min"), median=("median_rate", "first"), highest=("contract_unit_rate", "max"),
        excess_vs_median=("excess_vs_median", "sum")).reset_index()
    items["spread_pct"] = (100 * (items.highest / items.lowest - 1)).round(1)
    hi = pol.loc[pol.groupby(["item_code", "unit"]).contract_unit_rate.idxmax(), ["item_code", "unit", "project_id"]]
    items = items.merge(hi.rename(columns={"project_id": "highest_at"}), on=["item_code", "unit"])
    items["highest_at"] = items.highest_at.map(lambda p: labels.get(p, p))
    items = items.sort_values("excess_vs_median", ascending=False)
    idx = pol.groupby("project_id").apply(
        lambda d: pd.Series({"price_index": 100 * (d.contract_unit_rate * d.contract_qty).sum() / (d.median_rate * d.contract_qty).sum(),
                             "excess_vs_median": d.excess_vs_median.sum(), "items_compared": len(d)}),
        include_groups=False).reset_index()
    idx["project"] = idx.project_id.map(lambda p: labels.get(p, p))
    return items, idx.sort_values("price_index", ascending=False)


# ---------------------------------------------------------------- cash flow
def cash_view(tables, gates, exceptions, controls, projects, primes, end_period, prog_table):
    """Payments, overdue bills, retention held, advance recovery and next month's cash need, per project."""
    po = _scope_po(tables, projects, primes)
    dup = duplicate_ids(gates)
    inv = tables["invoices"][~tables["invoices"].invoice_id.isin(dup)]
    inv = inv[inv.po_id.isin(po.po_id) & (inv.period <= end_period)].merge(po[["po_id", "retainage_rate"]], on="po_id")
    pc = (controls or {}).get("project_controls")
    pay = (controls or {}).get("payments")
    pc_i = pc.set_index("project_id") if pc is not None else None
    risk = split_risk(gates, exceptions)
    rows, overdue_rows = [], []
    for p, g in inv.groupby("project_id"):
        g = g.sort_values(["period", "invoice_date"])
        terms = pc_i.payment_terms_days.get(p, np.nan) if pc_i is not None else np.nan
        dd = (pc_i.data_date.get(p) if pc_i is not None else pd.NaT)
        if pd.isna(dd):
            dd = pd.Period(end_period, "M").end_time.normalize()
        r = {"project_id": p, "billed_gross": g.gross_amount.sum(), "retention_held": g.retainage_withheld.sum(), "terms": terms}
        sov = prog_table.set_index("project_id").original_contract.get(p, np.nan)
        committed = prog_table.set_index("project_id").committed.get(p, np.nan)
        if pay is not None:
            pj = g[["invoice_id", "invoice_date", "net_amount", "gross_amount"]].merge(pay, on="invoice_id", how="left")
            pj["submitted"] = pj.submitted_date.fillna(pd.to_datetime(pj.invoice_date, errors="coerce"))
            paid = pj[pj.paid_date.notna() & (pj.paid_date <= dd)]
            r["paid_to_date"] = paid.amount_paid.sum()
            days = (paid.paid_date - paid.submitted).dt.days
            r["avg_days_to_pay"] = days.mean() if len(days) else np.nan
            r["paid_within_terms_pct"] = 100 * (days <= terms).mean() if len(days) and pd.notna(terms) else np.nan
            unpaid = pj[pj.paid_date.isna() | (pj.paid_date > dd)]
            r["unpaid_bills"] = len(unpaid)
            r["unpaid_amount"] = unpaid.net_amount.sum()
            age = (dd - unpaid.submitted).dt.days
            late = unpaid[age > terms] if pd.notna(terms) else unpaid.iloc[0:0]
            r["overdue_amount"] = late.net_amount.sum()
            for x in late.itertuples():
                overdue_rows.append({"project_id": p, "invoice_id": x.invoice_id, "submitted": x.submitted.date(),
                                     "days_waiting": int((dd - x.submitted).days), "net_amount": x.net_amount})
            # flagged money that was paid anyway (not withheld) and must be recovered from the next bill
            pr = paid.merge(risk, on="invoice_id", how="left").fillna({"blocked": 0, "to_check": 0})
            r["paid_needs_recovery"] = (pr.blocked + pr.to_check - pr.amount_withheld.fillna(0)).clip(lower=0).sum()
            r["advance_recovered"] = pj.advance_recovered.fillna(0).sum()
        if pc_i is not None and p in pc_i.index and pc_i.advance_paid.get(p, 0) > 0:
            adv = pc_i.at[p, "advance_paid"]
            a, b = pc_i.at[p, "advance_recovery_start_pct"], pc_i.at[p, "advance_recovery_end_pct"]
            prog = 100 * g.gross_amount.sum() / sov if sov else np.nan
            expected = adv * min(1, max(0, (prog - a) / (b - a))) if pd.notna(prog) and b > a else np.nan
            r.update({"advance_paid": adv, "progress_pct_of_contract": prog, "advance_expected_recovered": expected,
                      "advance_shortfall": max(0, expected - r.get("advance_recovered", 0)) if pd.notna(expected) else np.nan})
        # next month: average of the last 3 bills, capped by what is left on the contract
        recent = g.groupby("period").gross_amount.sum().iloc[-3:].mean()
        left = max(0.0, committed - g.gross_amount.sum()) if pd.notna(committed) else recent
        nxt = min(recent, left)
        ret = g.retainage_rate.iloc[0] if pd.notna(g.retainage_rate.iloc[0]) else 0
        adv_due = 0.0
        if "advance_paid" in r and sov:
            prog2 = 100 * (g.gross_amount.sum() + nxt) / sov
            a, b = pc_i.at[p, "advance_recovery_start_pct"], pc_i.at[p, "advance_recovery_end_pct"]
            exp2 = r["advance_paid"] * min(1, max(0, (prog2 - a) / (b - a)))
            adv_due = max(0.0, exp2 - r.get("advance_recovered", 0))
        r["next_month_gross"] = nxt
        r["next_month_net"] = max(0.0, nxt * (1 - ret) - adv_due)
        r["name"] = prog_table.set_index("project_id").name.get(p, p)
        rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(overdue_rows)


# ---------------------------------------------------------------- policy simulator
def policy_scenarios(run_fn, current_tol, scope_fn, truth=None):
    """Run the engine under a few tolerance policies and compare workload, money protected and accuracy."""
    zero = {k: 0.0 for k in current_tol if k.endswith(("_pct", "_pts")) and k != "money_tol"}
    scen = {
        "Strict: zero tolerance": {**current_tol, **zero},
        "Current settings": dict(current_tol),
        "Loose: double the current tolerances": {k: (v * 2 if k.endswith(("_pct", "_pts")) else v) for k, v in current_tol.items()},
        "Very loose: qty 10%, lump-sum 10 pts": {**current_tol, "qty_review_pct": 10.0, "qty_hold_pct": 15.0,
                                                          "pct_review_pts": 10.0, "pct_hold_pts": 15.0},
    }
    rows = []
    for name, tol in scen.items():
        tol["qty_hold_pct"] = max(tol["qty_hold_pct"], tol["qty_review_pct"])
        tol["pct_hold_pts"] = max(tol["pct_hold_pts"], tol["pct_review_pts"])
        ex, g = run_fn(tol)
        gf, ef = scope_fn(g, ex)
        rs = split_risk(gf, ef)
        r = {"policy": name, "bills_on_hold": int((gf.gate == "HOLD").sum()), "bills_to_review": int((gf.gate == "REVIEW").sum()),
             "first_pass_pct": round(100 * (gf.gate == "PASS").mean(), 0) if len(gf) else np.nan,
             "money_blocked": rs.blocked.sum(), "money_to_check": rs.to_check.sum()}
        if truth is not None and len(truth):
            def ks(df, col):
                return set(zip(df.invoice_id.astype(str), pd.to_numeric(df.line_no, errors="coerce").fillna(-1).astype(int), df[col]))
            det, tru = ks(ex, "rule"), ks(truth, "error_type")
            r["errors_caught"] = f"{len(det & tru)} of {len(tru)}"
            r["false_alarms"] = len(det - tru)
            ob = truth.assign(k=list(zip(truth.invoice_id.astype(str), pd.to_numeric(truth.line_no, errors="coerce").fillna(-1).astype(int), truth.error_type)))
            missed = ob[~ob.k.isin(det)]
            r["overbilling_missed"] = pd.to_numeric(missed.overbilled_amount, errors="coerce").fillna(0).sum()
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- manager insights
def manager_insights(pt, rc, perf, bench_idx, cash, overdue):
    out = []
    if len(pt) and pt.vac.notna().any():
        over = pt[pt.vac < 0].sort_values("vac")
        for r in over.head(2).itertuples():
            out.append({"level": "critical", "amount": -r.vac,
                        "title": f"{r.name} is forecast {money(-r.vac)} over budget",
                        "detail": f"Budget {money(r.budget)}; committed {money(r.committed)} plus {money(r.pending_co)} of pending change orders "
                                  f"= forecast {money(r.eac)}.",
                        "action": "Decide the pending change orders on value, find savings or request budget before more work is committed."})
    if len(pt) and pt.ld_exposure.fillna(0).sum() > 0:
        late = pt[pt.delay_weeks > 0].sort_values("ld_exposure", ascending=False)
        r = late.iloc[0]
        out.append({"level": "critical" if r.delay_weeks > 4 else "warning", "amount": r.ld_exposure,
                    "title": f"{r['name']} is {r.delay_weeks:.0f} week{'s' if round(r.delay_weeks) != 1 else ''} late: delay penalty exposure {money(r.ld_exposure)}",
                    "detail": f"{r.verified_pct:.0f}% complete vs {r.planned_pct:.0f}% planned (schedule index {r.spi:.2f}). "
                              f"Forecast finish {r.forecast_finish:%d %b %Y} vs contract {r.planned_finish:%d %b %Y}."
                              + (f" {len(late) - 1} other project(s) also late." if len(late) > 1 else ""),
                    "action": "Ask the contractor for a recovery plan (more crews, extra shifts) with weekly targets; "
                              "issue a formal delay notice to protect the right to the penalty."})
    if len(cash) and "paid_needs_recovery" in cash and cash.paid_needs_recovery.sum() > 0:
        c = cash[cash.paid_needs_recovery > 0].sort_values("paid_needs_recovery", ascending=False)
        out.append({"level": "critical", "amount": c.paid_needs_recovery.sum(),
                    "title": f"Recover {money(c.paid_needs_recovery.sum())} already paid on flagged bills",
                    "detail": f"These bills were paid in full before the checks ran; largest at {c.iloc[0]['name']} "
                              f"({money(c.iloc[0].paid_needs_recovery)}).",
                    "action": "Deduct the amounts from each contractor's next bill and record the recovery."})
    if len(cash) and "advance_shortfall" in cash and cash.advance_shortfall.fillna(0).max() > 1000:
        c = cash.sort_values("advance_shortfall", ascending=False).iloc[0]
        out.append({"level": "warning", "amount": c.advance_shortfall,
                    "title": f"{c['name']}: advance recovery {money(c.advance_shortfall)} behind schedule",
                    "detail": f"At {c.progress_pct_of_contract:.0f}% progress, {money(c.advance_expected_recovered)} of the "
                              f"{money(c.advance_paid)} advance should be recovered; {money(c.advance_recovered)} has been.",
                    "action": "Recover the shortfall from the next bill; the advance must be fully recovered by the agreed progress point."})
    if len(rc) >= 2:
        top2 = rc.iloc[:2]
        out.append({"level": "info", "amount": top2.money_at_risk.sum(),
                    "title": f"Fix 2 root causes to lift first-pass rate from {rc.attrs.get('first_pass_now', 0):.0f}% to {top2.first_pass_if_fixed.iloc[-1]:.0f}%",
                    "detail": f"'{top2.root_cause.iloc[0]}' and '{top2.root_cause.iloc[1]}' account for "
                              f"{int(top2.invoices_affected.sum())} failed-bill instances.",
                    "action": f"Owners: {top2.owner.iloc[0]} ({top2.process_fix.iloc[0].lower()}); "
                              f"{top2.owner.iloc[1]} ({top2.process_fix.iloc[1].lower()})."})
    if len(perf):
        bad = perf[perf.action == "Performance meeting"]
        if len(bad):
            r = bad.iloc[0]
            out.append({"level": "warning", "amount": r.blocked + r.to_check,
                        "title": f"Call a performance meeting with {r.contractor} (score {r.score:.0f}/100)",
                        "detail": f"First-pass {r.first_pass_pct:.0f}% ({r.trend.lower()}), {r.risk_pct_of_billed:.1f}% of billing at risk"
                                  + (f", schedule index {r.schedule_spi:.2f}" if pd.notna(r.schedule_spi) else "")
                                  + (f". Repeat issues: {r.repeat_issues}." if r.repeat_issues != "-" else "."),
                        "action": "Agree a corrective plan with dates; link the next bill's release to it."})
    if len(bench_idx) >= 2:
        r = bench_idx.iloc[0]
        if r.price_index >= 103:
            out.append({"level": "info", "amount": r.excess_vs_median,
                        "title": f"{r.project} pays {r.price_index - 100:.0f}% above the program median for the same items",
                        "detail": f"{int(r.items_compared)} shared items compared; {money(r.excess_vs_median)} above median prices.",
                        "action": "Use median rates as the ceiling for its change orders and for the next tender's estimate."})
    if len(cash) and "avg_days_to_pay" in cash and cash.avg_days_to_pay.notna().any():
        slow = cash[cash.avg_days_to_pay > cash.terms + 2] if "terms" in cash else cash.iloc[0:0]
        if len(slow):
            r = slow.sort_values("avg_days_to_pay", ascending=False).iloc[0]
            out.append({"level": "info", "amount": 0,
                        "title": f"Owner pays late: {r['name']} bills take {r.avg_days_to_pay:.0f} days vs {r.terms:.0f}-day terms",
                        "detail": f"{len(slow)} of {len(cash)} projects average beyond terms; only {r.paid_within_terms_pct:.0f}% of "
                                  f"{r['name']} bills were paid on time.",
                        "action": "Find the slow step (certification or fund release) and set a target per step; late payment "
                                  "gives contractors grounds for delay and interest claims."})
    if len(overdue):
        out.append({"level": "warning", "amount": overdue.net_amount.sum(),
                    "title": f"{len(overdue)} bill(s) unpaid beyond payment terms ({money(overdue.net_amount.sum())})",
                    "detail": f"Oldest has waited {overdue.days_waiting.max()} days.",
                    "action": "Clear certified amounts now; late payment weakens your position on delay penalties and claims."})
    return out
