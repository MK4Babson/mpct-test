"""
Schedule analytics: CPM, PERT, schedule risk (Monte Carlo), crashing, time impact analysis and schedule quality checks.

Each building gets a task network (template by structure type: data center or canal). Durations are fitted so the
baseline finishes on the contract's planned finish. Progress comes from the field-verified quantities of the contract
items mapped to each task, so the CPM forecast is driven by the same data as the bill checks.

Methods
  CPM      forward / backward pass, early/late start and finish, total and free float, critical path
  PERT     three-point estimates, te = (o + 4m + p) / 6, sd = (p - o) / 6, P(finish by date) on the critical path
  SRA      Monte Carlo with beta-PERT durations: finish distribution, criticality index, merge bias
  Crashing greedy time-cost trade-off on the critical path (cheapest $/day first), vs the daily delay cost
  TIA      time impact analysis: insert a delay on one task, re-run the forecast, read the change in finish
  Quality  DCMA-style checks: open ends, high float, long durations, negative float against the contract date

Durations are calendar days. Templates and crash costs are illustrative (synthetic) and labelled as such in the app.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

# id, name, trade, predecessors, (optimistic, most likely, pessimistic) days, item codes (progress), kind, crash $/day, crash max days
DATACENTER = [
    ("A01", "Mobilize and clear site", "Civil", [], (7, 10, 15), ["C-LS1", "C-LS2", "E-LS1", "M-LS1"], "work", 8000, 3),
    ("A02", "Mass excavation", "Civil", ["A01"], (15, 20, 30), ["C-301"], "work", 12000, 6),
    ("A03", "Underground storm drain", "Civil", ["A02"], (15, 20, 30), ["C-305"], "work", 10000, 5),
    ("A04", "Duct banks", "Civil", ["A02"], (15, 20, 28), ["C-306"], "work", 10000, 5),
    ("A05", "Foundations and slab (concrete, rebar)", "Civil", ["A03"], (25, 35, 50), ["C-302", "C-303"], "work", 18000, 8),
    ("A06", "Aggregate base and paving", "Civil", ["A04", "A05"], (10, 15, 22), ["C-304"], "work", 8000, 4),
    ("A07", "Building shell: steel, roof, enclosure (by others)", "Shell", ["A05"], (40, 50, 70), [], "time", 25000, 10),
    ("A08", "Equipment yard pads", "Civil", ["A04"], (8, 10, 15), [], "time", 6000, 3),
    ("A09", "Conduit runs", "Electrical", ["A04"], (15, 20, 30), ["E-103"], "work", 9000, 5),
    ("A10", "Cable tray", "Electrical", ["A07"], (20, 25, 35), ["E-101"], "work", 12000, 6),
    ("A11", "MV cable pull", "Electrical", ["A09", "A10"], (15, 20, 30), ["E-102"], "work", 14000, 5),
    ("A12", "Busway", "Electrical", ["A07"], (15, 20, 28), ["E-104"], "work", 10000, 5),
    ("A13", "Switchgear manufacture and delivery (long lead)", "Electrical", ["A01"], (120, 150, 210), [], "delivery", 20000, 20),
    ("A14", "Switchgear set and terminate", "Electrical", ["A13", "A08", "A11"], (10, 14, 20), ["E-105"], "work", 15000, 4),
    ("A15", "UPS set and terminate", "Electrical", ["A14", "A12"], (10, 14, 20), ["E-106"], "work", 15000, 4),
    ("A16", "Generator manufacture and delivery (long lead)", "Electrical", ["A01"], (140, 170, 230), [], "delivery", 18000, 20),
    ("A17", "Generators set and terminate", "Electrical", ["A16", "A08"], (8, 12, 18), ["E-107"], "work", 12000, 3),
    ("A18", "Chiller manufacture and delivery", "Mechanical", ["A01"], (110, 140, 190), [], "delivery", 15000, 15),
    ("A19", "Chillers set and connect", "Mechanical", ["A18", "A08"], (8, 12, 18), ["M-204"], "work", 12000, 3),
    ("A20", "Chilled water piping", "Mechanical", ["A07"], (25, 35, 50), ["M-201"], "work", 14000, 8),
    ("A21", "Process (coolant) piping", "Mechanical", ["A07"], (20, 28, 40), ["M-202"], "work", 12000, 6),
    ("A22", "Ductwork", "Mechanical", ["A07"], (15, 20, 30), ["M-206"], "work", 9000, 5),
    ("A23", "CRAH units set and connect", "Mechanical", ["A22", "A20"], (8, 12, 18), ["M-205"], "work", 10000, 3),
    ("A24", "CDU set and connect", "Mechanical", ["A21", "A20"], (8, 12, 18), ["M-203"], "work", 10000, 3),
    ("A25", "Energize permanent power", "Electrical", ["A14", "A15", "A17"], (3, 5, 10), [], "work", 20000, 1),
    ("A26", "Commissioning L3-L4 (functional tests)", "Commissioning", ["A25", "A19", "A23", "A24"], (15, 20, 30), ["E-LS2", "M-LS2"], "work", 15000, 5),
    ("A27", "Integrated systems test (L5)", "Commissioning", ["A26"], (7, 10, 15), [], "work", 15000, 2),
    ("A28", "Substantial completion", "Milestone", ["A27", "A06"], (0, 0, 0), [], "milestone", 0, 0),
]
CANAL = [
    ("K01", "Mobilize, survey and set out", "Preliminaries", [], (10, 15, 25), ["A1", "A2", "A3", "E2", "E3"], "work", 1500, 4),
    ("K02", "Clearing and dismantling old lining", "Preliminaries", ["K01"], (20, 30, 45), ["A4", "A5"], "work", 2000, 8),
    ("K03", "Earthwork: excavation and filling", "Earthwork", ["K02"], (60, 80, 115), ["B1", "B2"], "work", 4000, 20),
    ("K04", "Bank restoration, CNS and sand layers", "Earthwork", ["K03"], (30, 40, 60), ["B3", "B4", "B5"], "work", 3000, 10),
    ("K05", "Under-drains and pressure relief", "Lining", ["K04"], (15, 20, 35), ["C1", "C2", "C3"], "work", 2000, 5),
    ("K06", "LDPE film and concrete lining", "Lining", ["K05"], (60, 80, 120), ["C4", "C5", "C6", "C7", "C8"], "work", 5000, 20),
    ("K07", "Drains, toe drains and pitching", "Lining", ["K04"], (20, 30, 45), ["C9", "C10", "C11"], "work", 1500, 8),
    ("K08", "Falls, regulators, outlets and CD works", "Structures", ["K02"], (90, 120, 170), ["D-FALL", "D-HR", "D-HRP", "D-INL", "D-OUT", "D-CD"], "work", 3500, 25),
    ("K09", "Bridges and culverts", "Structures", ["K03"], (60, 80, 120), ["D-SLAB", "D-PIPE", "E1"], "work", 3000, 20),
    ("K10", "Trial water release and handover", "Commissioning", ["K06", "K07", "K08", "K09"], (10, 15, 25), ["E4"], "work", 2000, 4),
    ("K11", "Completion", "Milestone", ["K10"], (0, 0, 0), [], "milestone", 0, 0),
]
TEMPLATES = {"datacenter": DATACENTER, "canal": CANAL}
COLS = ["id", "name", "trade", "preds", "opt_m_pes", "items", "kind", "crash_cost", "crash_max"]


# ---------------------------------------------------------------- network building
def template(structure):
    t = pd.DataFrame(TEMPLATES.get(structure, DATACENTER), columns=COLS)
    t[["o", "m", "p"]] = pd.DataFrame(t.opt_m_pes.tolist(), index=t.index).astype(float)
    return t.drop(columns="opt_m_pes")


def topo_order(acts):
    preds = dict(zip(acts.id, acts.preds))
    order, seen = [], set()

    def visit(a, stack=()):
        if a in seen:
            return
        if a in stack:
            raise ValueError(f"Loop in the network at {a}")
        for p in preds.get(a, []):
            visit(p, stack + (a,))
        seen.add(a)
        order.append(a)
    for a in acts.id:
        visit(a)
    return order


def cpm(acts, dur_col="dur", start_col=None, finish_by=None):
    """Forward and backward pass. Returns a copy with es, ef, ls, lf, total_float, free_float, critical.
    start_col: optional earliest start (e.g. the data date for unfinished tasks). finish_by: late finish for the
    last task (e.g. the contract date: negative float means late); default = the project's own early finish."""
    a = acts.set_index("id")
    order = topo_order(acts)
    succ = defaultdict(list)
    for i, ps in a.preds.items():
        for p in ps:
            succ[p].append(i)
    es, ef = {}, {}
    for i in order:
        s = max([ef[p] for p in a.preds[i]], default=0.0)
        if start_col:
            s = max(s, a.at[i, start_col])
        es[i], ef[i] = s, s + a.at[i, dur_col]
    finish = max(ef.values())
    end = finish if finish_by is None else finish_by
    ls, lf = {}, {}
    for i in reversed(order):
        f = min([ls[s] for s in succ[i]], default=end)
        lf[i], ls[i] = f, f - a.at[i, dur_col]
    out = acts.copy()
    out["es"], out["ef"] = out.id.map(es), out.id.map(ef)
    out["ls"], out["lf"] = out.id.map(ls), out.id.map(lf)
    out["total_float"] = out.ls - out.es
    out["free_float"] = [min([es[s] for s in succ[i]], default=end) - ef[i] for i in out.id]
    tf_min = out.total_float.min()
    out["critical"] = out.total_float <= tf_min + 0.5
    out.attrs["finish"] = finish
    return out


def critical_path(c):
    return list(c[c.critical].sort_values("es").id)


def item_progress(tables, project_id, end_period):
    """Verified % complete per contract item (value-weighted), from field measurements up to end_period."""
    pol, po, grn = tables["po_lines"], tables["purchase_orders"], tables["goods_receipts"]
    pos = po[po.project_id == project_id].po_id
    lines = pol[pol.po_id.isin(pos) & (pol.line_source == "CONTRACT")].copy()
    g = grn[grn.po_id.isin(pos) & (grn.period.astype(str) <= end_period)]
    qty = g.groupby(["po_id", "item_code"]).verified_qty.sum()
    pct_ls = g.dropna(subset=["verified_pct_complete_cum"]).sort_values("period").groupby(["po_id", "item_code"]).verified_pct_complete_cum.last()
    out = []
    for r in lines.itertuples():
        k = (r.po_id, r.item_code)
        if str(r.unit).upper() == "LS" or (pd.isna(r.contract_qty) or not r.contract_qty):
            pct = float(pct_ls.get(k, 0.0))
        else:
            pct = 100 * float(qty.get(k, 0.0)) / float(r.contract_qty)
        out.append({"item_code": r.item_code, "value": float(r.scheduled_value or 0), "pct": min(100.0, max(0.0, pct))})
    return pd.DataFrame(out)


def _match(code, keys):
    return any(code == k or (k.startswith("D-") and code.startswith(k + "-")) for k in keys)


def build(project, tables, controls_row, end_period, structure="datacenter"):
    """Network for one project: fitted durations, progress from verified quantities, forecast from the data date."""
    t = template(structure)
    start = pd.Timestamp(controls_row["planned_start"])
    pfin = pd.Timestamp(controls_row["planned_finish"])
    data_date = pd.Timestamp(controls_row.get("data_date") or pd.Period(end_period, "M").end_time.normalize())
    data_date = min(data_date, pd.Period(end_period, "M").end_time.normalize())
    base = cpm(t.assign(dur=t.m))
    k = max(1.0, (pfin - start).days) / max(1.0, base.attrs["finish"])  # fit the template to the contract window
    for c in ["o", "m", "p"]:
        t[c] = t[c] * k
    t["dur"] = t.m
    t["crash_max"] = (t.crash_max * k).round()
    # progress from field-verified quantities
    ip = item_progress(tables, project, end_period)
    pct, value = [], []
    for r in t.itertuples():
        m = ip[ip.item_code.map(lambda c: _match(c, r.items))] if len(ip) and r.items else ip.iloc[0:0]
        v = m.value.sum()
        pct.append(float((m.pct * m.value).sum() / v) if v > 0 else np.nan)
        value.append(v)
    t["value"] = value
    t["pct_verified"] = pct
    baseline = cpm(t)
    t["bl_start"] = baseline.es
    t["bl_finish"] = baseline.ef
    dd = (data_date - start).days
    # tasks with no contract items (deliveries, shell by others): assume on plan, by elapsed time
    time_pct = ((dd - t.bl_start) / t.dur.replace(0, np.nan) * 100).clip(0, 100).fillna(pd.Series(np.where(t.bl_finish <= dd, 100.0, 0.0), index=t.index))
    t["pct"] = np.where(t.pct_verified.notna(), t.pct_verified, time_pct)
    t["pct_source"] = np.where(t.pct_verified.notna(), "field-verified quantities", "assumed on plan (no item data)")
    t.loc[t.kind == "milestone", "pct"] = 0.0
    t["remaining"] = t.dur * (1 - t.pct / 100)
    t["started"] = t.pct > 0.5
    t["done"] = t.pct >= 99.5
    contract_days = float((pfin - start).days)
    meta = {"start": start, "planned_finish": pfin, "data_date": data_date, "data_day": float(dd), "contract_days": contract_days,
            "fit_factor": k, "structure": structure}
    return t, meta


def forecast(t, meta, extra=None):
    """Re-schedule the unfinished work from the data date (progress override for started tasks)."""
    f = t.copy()
    f["fdur"] = np.where(f.done, 0.0, f.remaining)
    if extra:
        for a, d in extra.items():
            f.loc[f.id == a, "fdur"] += d
    dd = meta["data_day"]
    f["min_start"] = np.where(f.done, 0.0, dd)
    c = cpm(f, "fdur", "min_start", finish_by=meta["contract_days"])
    c["fstart"], c["ffinish"] = c.es, c.ef
    c.loc[c.done, ["fstart", "ffinish"]] = np.nan
    c.loc[c.done, "critical"] = False
    return c


def dates(meta, days):
    return meta["start"] + pd.to_timedelta(np.asarray(days, float), unit="D")


# ---------------------------------------------------------------- PERT
def pert(t, meta):
    p = t.copy()
    p["te"] = (p.o + 4 * p.m + p.p) / 6 * (1 - p.pct / 100)
    p["sd"] = (p.p - p.o) / 6 * (1 - p.pct / 100)
    p.loc[p.done, ["te", "sd"]] = 0.0
    p["min_start"] = np.where(p.done, 0.0, meta["data_day"])
    c = cpm(p, "te", "min_start", finish_by=meta["contract_days"])
    cp = c[c.critical]
    mu = c.attrs["finish"]
    sd = float(np.sqrt((cp.sd ** 2).sum()))
    prob = float(stats.norm.cdf((meta["contract_days"] - mu) / sd)) if sd > 0 else float(mu <= meta["contract_days"])
    return {"table": c, "te_finish": mu, "sd": sd, "p_on_time": prob, "path": critical_path(c)}


def p_by(pr, day):
    return float(stats.norm.cdf((day - pr["te_finish"]) / pr["sd"])) if pr["sd"] > 0 else float(pr["te_finish"] <= day)


# ---------------------------------------------------------------- Monte Carlo schedule risk
def _beta_pert(o, m, p, size, rng):
    o, m, p = map(np.asarray, (o, m, p))
    rngw = np.where(p > o, p - o, 1.0)
    a = 1 + 4 * (m - o) / rngw
    b = 1 + 4 * (p - m) / rngw
    x = rng.beta(a, b, size=(size, len(o)))
    return np.where(p > o, o + x * (p - o), m)


def monte_carlo(t, meta, n=3000, seed=11):
    rng = np.random.default_rng(seed)
    ids = list(t.id)
    idx = {a: i for i, a in enumerate(ids)}
    rem = 1 - t.pct.values / 100
    D = _beta_pert(t.o.values, t.m.values, t.p.values, n, rng) * rem
    D[:, t.done.values] = 0.0
    dd = meta["data_day"]
    mins = np.where(t.done.values, 0.0, dd)
    order = topo_order(t)
    preds = {a: [idx[p] for p in ps] for a, ps in zip(t.id, t.preds)}
    ES = np.zeros((n, len(ids)))
    EF = np.zeros((n, len(ids)))
    for a in order:
        i = idx[a]
        s = np.max(EF[:, preds[a]], axis=1) if preds[a] else np.zeros(n)
        s = np.maximum(s, mins[i])
        ES[:, i], EF[:, i] = s, s + D[:, i]
    finish = EF.max(axis=1)
    succ = defaultdict(list)
    for a, ps in zip(t.id, t.preds):
        for p in ps:
            succ[idx[p]].append(idx[a])
    LF = np.zeros_like(EF)
    LS = np.zeros_like(EF)
    for a in reversed(order):
        i = idx[a]
        lf = np.min(LS[:, succ[i]], axis=1) if succ[i] else finish
        LF[:, i], LS[:, i] = lf, lf - D[:, i]
    crit = (LS - ES) <= 0.5
    ci = pd.Series(crit.mean(axis=0), index=ids)
    ci[t.set_index("id").done] = 0.0
    det = pert(t, meta)["te_finish"]
    return {"finish": finish, "criticality": ci, "p50": float(np.percentile(finish, 50)), "p80": float(np.percentile(finish, 80)),
            "p_on_time": float((finish <= meta["contract_days"]).mean()), "merge_bias": float(finish.mean() - det), "pert_finish": det}


# ---------------------------------------------------------------- crashing
def crash(t, meta, days_wanted, delay_cost_per_day):
    """Time-cost trade-off: each step buys back about one day at the lowest cost per day saved. Tries single tasks on
    the critical path and, where parallel critical paths need it, pairs of tasks (one on each path)."""
    from itertools import combinations
    f = t.copy()
    left = dict(zip(f.id, np.where(f.done | (f.kind == "milestone"), 0.0, np.minimum(f.crash_max, (f.remaining * 0.4).round()))))
    cost = dict(zip(f.id, f.crash_cost))
    name = dict(zip(f.id, f.name))
    cut = {a: 0.0 for a in f.id}
    run = lambda: forecast(f, meta, {k: -d for k, d in cut.items() if d})  # noqa: E731
    cur = run()
    base = cur.attrs["finish"]
    steps, total = [], 0.0
    for _ in range(int(days_wanted)):
        cand = [a for a in cur[cur.critical].id if left[a] - cut[a] >= 1]
        cand = sorted(cand, key=lambda a: cost[a])[:7]
        if not cand:
            break
        options = [(a,) for a in cand] + list(combinations(cand, 2))
        best = None
        for opt in options:
            for a in opt:
                cut[a] += 1
            fin = run().attrs["finish"]
            for a in opt:
                cut[a] -= 1
            saved = cur.attrs["finish"] - fin
            if saved < 0.5:
                continue
            c = sum(cost[a] for a in opt)
            ratio = c / saved
            if best is None or ratio < best[0]:
                best = (ratio, opt, c, saved)
        if best is None:
            break
        ratio, opt, c, saved = best
        for a in opt:
            cut[a] += 1
        cur = run()
        total += c
        steps.append({"step": len(steps) + 1, "task": " + ".join(name[a] for a in opt), "cost": c, "days_saved": saved,
                      "cost_per_day": ratio, "cumulative_cost": total, "days_saved_total": base - cur.attrs["finish"],
                      "worth_it": ratio <= delay_cost_per_day})
    s = pd.DataFrame(steps)
    s.attrs["base_finish"] = base
    return s


# ---------------------------------------------------------------- time impact analysis
def tia(t, meta, task_id, days):
    before = forecast(t, meta).attrs["finish"]
    after = forecast(t, meta, {task_id: float(days)}).attrs["finish"]
    return max(0.0, after - before)


ITEM_TASK = {code: tid for tid, _, _, _, _, items, *_ in DATACENTER + CANAL for code in items}


def task_for_item(code):
    if code in ITEM_TASK:
        return ITEM_TASK[code]
    for k, v in ITEM_TASK.items():
        if k.startswith("D-") and str(code).startswith(k + "-"):
            return v
    return None


# ---------------------------------------------------------------- quality checks
def quality(t, f, meta):
    succ = defaultdict(list)
    for a, ps in zip(t.id, t.preds):
        for p in ps:
            succ[p].append(a)
    first, last = t.id.iloc[0], t.id.iloc[-1]
    open_start = [a for a, ps in zip(t.id, t.preds) if not ps and a != first]
    open_end = [a for a in t.id if not succ[a] and a != last]
    work = t[t.kind == "work"]
    long_d = list(work[work.dur > 44].name)
    hi = f[(f.total_float > 44) & ~f.done]
    neg = f[(f.total_float < 0) & ~f.done]
    checks = [
        ("Logic: every task has a predecessor and a successor", not open_start and not open_end,
         "none open" if not open_start and not open_end else f"open ends: {', '.join(open_start + open_end)}"),
        ("Long durations (> 44 days, work tasks)", len(long_d) <= max(1, len(work) // 4),
         f"{len(long_d)} of {len(work)}: " + (", ".join(long_d[:4]) if long_d else "none") + " (split long tasks to track them)"),
        ("High float (> 44 days)", len(hi) <= len(f) // 5, f"{len(hi)} tasks: " + (", ".join(hi.name.head(4)) if len(hi) else "none") + " (often a missing link)"),
        ("Negative float against the contract date", len(neg) == 0,
         f"{len(neg)} tasks behind the contract date, worst {neg.total_float.min():.0f} days" if len(neg) else "none: on track for the contract date"),
        ("Critical path runs start to finish", f.critical.any(), " → ".join(critical_path(f)[:3]) + " … " if f.critical.any() else "no critical path"),
    ]
    return pd.DataFrame(checks, columns=["check", "pass", "detail"])
