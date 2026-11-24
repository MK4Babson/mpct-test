"""
Materials and supply-chain engine.

One stock ledger drives everything. Every movement (receipt, transfer, issue, adjustment) is one row; balances at the
hub, sub-stores and site stores are sums of those rows, so one bag of cement moving anywhere updates every view.

  Ledger      unit conversion to the base unit (e.g. bags -> tonnes), balances by location, balance history
  MRP         gross requirement from the CPM forecast (remaining material spread over each task's forecast window),
              netted against stock and open orders, offset by lead time; late releases become expedite actions
  Lookahead   six-week constraint list (Last Planner style): tasks starting soon whose material is not secured
  Policy      ABC (usage value) x XYZ (demand variability), EOQ, safety stock z*sqrt(L*sd_d^2 + d^2*sd_L^2), reorder point,
              risk pooling (one pooled buffer at the hub vs one per site)
  Transfers   min-cost transshipment LP (SciPy HiGHS) from stores with surplus to sites with shortages, with shadow prices
  Cash        value on hand, carrying cost per day, weeks of supply, cash tied up in excess stock, buy-early vs buy-late
  Suppliers   OTIF (on time AND in full), actual lead time vs master data
  Forecast    WAPE, bias and MASE of the site's consumption plan against actual issues
  Data        master-data and ledger checks (duplicates, unit errors, negative stock, stale lead times)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data" / "scm"
NAMES = ["items", "locations", "suppliers", "ledger", "deliveries", "open_pos", "plan_weekly", "baseline_windows", "fx_at_order",
         "factories", "lanes", "tier2", "shipments"]
STORE_TIERS = ("hub", "sub-store", "site")
CAPITAL_RATE = 0.08            # owner's cost of capital for carrying cost (editable in the app)
ORDER_COST = {"USA": 250.0}    # $ per purchase order; imports cost more to place (docs, broker, inspection)
IMPORT_ORDER_COST = 1500.0
TRUCK_COST_PER_MILE = 3.5
UNITS_PER_TRUCK = {"CEM-OPC": 25, "AGG-57": 25, "RBR-5": 22, "STL-W": 20, "CT-24": 4000, "MV-500": 10000, "CND-4": 8000,
                   "LV-CU": 60000, "BUS-4K": 400, "PIPE-CHW": 2000, "DUCT": 15000}
RACI = {"Concrete": ("Site material coordinator", "Construction manager"), "Steel": ("Material planner", "Construction manager"),
        "Electrical": ("Electrical prime's planner", "Owner's PM"), "Mechanical": ("Mechanical prime's planner", "Owner's PM"),
        "Equipment": ("Owner's procurement lead", "Owner's PM")}


# ---------------------------------------------------------------- loading
def load(folder=FOLDER):
    d = {n: pd.read_csv(folder / f"{n}.csv") for n in NAMES if (folder / f"{n}.csv").exists()}
    for n, cols in {"ledger": ["date"], "deliveries": ["order_date", "promised_date", "delivered_date"], "open_pos": ["due_date", "pay_date"],
                    "plan_weekly": ["week"], "baseline_windows": ["bl_start", "bl_finish"], "shipments": ["promised_due"]}.items():
        if n in d:
            for c in cols:
                d[n][c] = pd.to_datetime(d[n][c])
    for n in ("ledger", "open_pos", "deliveries"):
        if n in d:
            for c in ("project_id", "ref", "note", "from_loc", "to_loc"):
                if c in d[n]:
                    d[n][c] = d[n][c].fillna("")
    d["items"]["alt_unit"] = d["items"]["alt_unit"].fillna("")
    return d


def to_base(ledger, items):
    """Convert every row to the item's base unit. Unknown units are flagged, not guessed."""
    it = items.set_index("item_id")
    led = ledger.copy()
    base = led.item_id.map(it.unit)
    alt = led.item_id.map(it.alt_unit)
    per = led.item_id.map(it.alt_per_unit)
    led["qty_base"] = np.where(led.unit == base, led.qty, np.where((led.unit == alt) & (alt != ""), led.qty / per, np.nan))
    led["unit_ok"] = led.qty_base.notna()
    led["base_unit"] = base
    return led


def suspect_rows(led):
    """Rows the data checks flag: duplicated transaction IDs (second copy) and issues far above normal (unit errors)."""
    dup = led.duplicated("txn_id", keep="first")
    iss = (led.txn_type == "ISSUE") & led.unit_ok
    med = led[iss].groupby(["item_id", "project_id"]).qty_base.transform("median")
    big = pd.Series(False, index=led.index)
    big.loc[med.index] = led.loc[med.index, "qty_base"] > 6 * med
    return dup | big


def clean(led):
    """Ledger for planning analytics: drop rows the data checks flag (they stay in the ledger and on the Data tab)."""
    return led[~suspect_rows(led)]


def movements(led, stores):
    """Signed movements per store: +qty into to_loc, -qty out of from_loc (suppliers and consumption are outside)."""
    a = led[led.to_loc.isin(stores)][["date", "txn_id", "item_id", "to_loc", "qty_base", "txn_type"]].rename(columns={"to_loc": "location_id"})
    b = led[led.from_loc.isin(stores)][["date", "txn_id", "item_id", "from_loc", "qty_base", "txn_type"]].rename(columns={"from_loc": "location_id"})
    b["qty_base"] = -b.qty_base
    return pd.concat([a, b]).sort_values(["date", "txn_id"])


def balances(led, locations, items, as_of=None):
    stores = set(locations.location_id)
    m = movements(led[led.unit_ok] if "unit_ok" in led else led, stores)
    if as_of is not None:
        m = m[m.date <= as_of]
    b = m.groupby(["location_id", "item_id"]).qty_base.sum().rename("on_hand").reset_index()
    b = b.merge(locations[["location_id", "name", "tier", "campus", "project_id"]], on="location_id", how="left")
    b = b.merge(items[["item_id", "name", "unit", "unit_cost_usd", "family"]].rename(columns={"name": "item_name"}), on="item_id", how="left")
    b["value_usd"] = b.on_hand * b.unit_cost_usd
    return b


def history(led, locations, item, location=None):
    m = movements(led[led.unit_ok], set(locations.location_id))
    m = m[m.item_id == item]
    if location:
        m = m[m.location_id == location]
    s = m.groupby("date").qty_base.sum().cumsum()
    return s.rename("on_hand").reset_index()


def post(ledger, date, item, frm, to, qty, unit, kind, project="", ref="manual entry"):
    """Append one movement (the app uses this for the 'move one bag' demo)."""
    n = len(ledger) + 1
    row = {"txn_id": f"M{n:06d}", "date": pd.Timestamp(date), "item_id": item, "from_loc": frm, "to_loc": to, "qty": float(qty), "unit": unit,
           "txn_type": kind, "project_id": project, "ref": ref, "note": "posted in app"}
    return pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)


# ---------------------------------------------------------------- requirements (MRP from the schedule)
def issued(led, data_date):
    x = led[(led.txn_type == "ISSUE") & (led.date <= data_date) & led.unit_ok]
    return x.groupby(["project_id", "item_id"]).qty_base.sum()


def requirements(items, projects, windows, iss, data_date, horizon_weeks=26):
    """Weekly gross requirement per project and item.

    windows: project_id, task_id, start, finish (dates of the CPM forecast; done tasks have finish <= data date)
    Remaining quantity = total for the building - already issued; spread evenly over the task's remaining forecast window.
    """
    scale = dict(zip(projects.project_id, projects.scale))
    weeks = pd.date_range(data_date + pd.Timedelta(days=1), periods=horizon_weeks, freq="W-MON")
    rows = []
    w = windows.set_index(["project_id", "task_id"])
    for it in items.itertuples():
        for p in projects.project_id:
            if (p, it.task_id) not in w.index:
                continue
            tw = w.loc[(p, it.task_id)]
            total = it.qty_per_building * scale.get(p, 1.0)
            if it.unit == "ea":
                total = float(max(1, round(total)))
            rem = max(0.0, total - float(iss.get((p, it.item_id), 0.0)))
            if rem <= 1e-6 * max(total, 1):
                continue
            s, f = max(pd.Timestamp(tw.start), data_date + pd.Timedelta(days=1)), pd.Timestamp(tw.finish)
            catch_up = f <= data_date + pd.Timedelta(days=1)
            if catch_up:   # task due to be finished but material not all used: catch-up over the next two weeks
                s, f = data_date + pd.Timedelta(days=1), data_date + pd.Timedelta(days=15)
            if f <= s:
                f = s + pd.Timedelta(days=7)
            span = (f - s).days
            for wk in weeks:
                ov = max(0, (min(wk + pd.Timedelta(days=7), f) - max(wk, s)).days)
                if ov > 0:
                    rows.append({"week": wk, "project_id": p, "item_id": it.item_id, "task_id": it.task_id, "gross": rem * ov / span,
                                 "catch_up": catch_up})
    r = pd.DataFrame(rows, columns=["week", "project_id", "item_id", "task_id", "gross", "catch_up"])
    return r


def mrp(items, req, bal, open_pos, policy, data_date, horizon_weeks=26):
    """Network-level MRP per item: projected available balance, net requirement, planned receipts and releases.

    Available = stock at every store (hub + sub-stores + sites). Scheduled receipts = open purchase orders by due date.
    Planned receipt = lot (EOQ rounded up to the supplier lot) when the projected balance would fall below safety stock.
    Planned release = planned receipt week - lead time. A release date in the past means the order is late -> expedite.
    """
    weeks = pd.date_range(data_date + pd.Timedelta(days=1), periods=horizon_weeks, freq="W-MON")
    pol = policy.set_index("item_id")
    out = []
    for it in items[items.family != "Equipment"].itertuples():
        if it.item_id not in pol.index:
            continue
        g = req[req.item_id == it.item_id].groupby("week").gross.sum().reindex(weeks, fill_value=0.0)
        if g.sum() <= 0:
            continue
        sr = open_pos[open_pos.item_id == it.item_id].drop_duplicates("po_id")
        sr = sr.assign(week=sr.due_date.dt.to_period("W-SUN").dt.start_time).groupby("week").qty.sum()
        sr = sr.reindex(weeks, fill_value=0.0) + sr[sr.index < weeks[0]].sum() * (weeks == weeks[0])
        oh = float(bal[bal.item_id == it.item_id].on_hand.sum())
        ss = float(pol.safety_stock.get(it.item_id, 0.0))
        lot = float(pol.order_qty.get(it.item_id, it.lot_size))
        lead = float(pol.lead_mean.get(it.item_id, it.lead_days))
        pab = oh
        for wk in weeks:
            pab = pab + sr[wk] - g[wk]
            planned = 0.0
            if pab < ss - 1e-9:
                short = ss - pab
                planned = short if lot <= 0 else np.ceil(short / lot) * lot
                pab += planned
            rel = wk - pd.Timedelta(days=lead) if planned else pd.NaT
            out.append({"item_id": it.item_id, "week": wk, "gross": g[wk], "scheduled_receipts": sr[wk], "planned_receipt": planned,
                        "projected_on_hand": pab, "safety_stock": ss, "planned_release": rel,
                        "status": ("Call-off LATE vs supplier lead time: confirm they can meet it" if planned and it.flow == "direct" and rel < data_date else
                                   "Call-off on blanket order" if planned and it.flow == "direct" else
                                   "EXPEDITE: release date already passed" if planned and rel < data_date else
                                   "Release order" if planned and rel <= data_date + pd.Timedelta(days=14) else
                                   "Planned" if planned else "")})
    return pd.DataFrame(out)


def lookahead(items, req, bal, open_pos, windows, locations, data_date, days=42):
    """Six-week constraint check per project and task: is the material for work starting soon secured?"""
    end = data_date + pd.Timedelta(days=days)
    sites = locations[locations.tier == "site"]
    campus_of = dict(zip(sites.project_id, sites.campus))
    r = req[(req.week <= end)].groupby(["project_id", "item_id", "task_id"]).gross.sum().reset_index()
    it = items.set_index("item_id")
    rows = []
    w = windows.set_index(["project_id", "task_id"])
    for x in r.itertuples():
        site = bal[(bal.location_id == f"SITE-{x.project_id}") & (bal.item_id == x.item_id)].on_hand.sum()
        camp = campus_of.get(x.project_id, "")
        campus = bal[(bal.campus == camp) & (bal.tier == "sub-store") & (bal.item_id == x.item_id)].on_hand.sum()
        due = open_pos[(open_pos.item_id == x.item_id) & ((open_pos.project_id == x.project_id) | (open_pos.project_id == "")) & (open_pos.due_date <= end)]
        incoming = due.drop_duplicates("po_id").qty.sum()
        fam = it.family[x.item_id]
        avail = site + (campus if fam != "Equipment" else 0.0) + (incoming if fam == "Equipment" else 0.0)
        short = max(0.0, x.gross - avail)
        start = w.start.get((x.project_id, x.task_id), pd.NaT)
        resp, acc = RACI.get(fam, ("Material planner", "Owner's PM"))
        rows.append({"project_id": x.project_id, "task_id": x.task_id, "task_start": start, "item_id": x.item_id, "item": it.name[x.item_id],
                     "unit": it.unit[x.item_id], "need_6wk": x.gross, "at_site": site, "at_campus_store": campus, "incoming_by_then": incoming,
                     "shortfall": short, "status": "CONSTRAINT" if short > 1e-6 else "Ready", "responsible": resp, "accountable": acc})
    return pd.DataFrame(rows).sort_values(["status", "task_start"])


# ---------------------------------------------------------------- inventory policy
def lead_stats(items, deliveries):
    d = deliveries.copy()
    d["lead_actual"] = (d.delivered_date - d.order_date).dt.days
    g = d.groupby("item_id").lead_actual.agg(["mean", "std", "count"])
    m = items.set_index("item_id")
    out = pd.DataFrame({"lead_mean": g["mean"].reindex(m.index).fillna(m.lead_days), "lead_sd": g["std"].reindex(m.index).fillna(m.lead_sd),
                        "lead_obs": g["count"].reindex(m.index).fillna(0).astype(int)})
    out["lead_master"] = m.lead_days
    return out


def forecast_errors(led, plan_weekly, data_date):
    """Weekly actual issues vs the site's plan, per item and project, up to the data date."""
    a = led[(led.txn_type == "ISSUE") & led.unit_ok & (led.date <= data_date)].copy()
    a["week"] = a.date.dt.to_period("W-SUN").dt.start_time
    act = a.groupby(["item_id", "project_id", "week"]).qty_base.sum()
    pl = plan_weekly[plan_weekly.week <= data_date].groupby(["item_id", "project_id", "week"]).planned_qty.sum()
    j = pd.concat([act.rename("actual"), pl.rename("plan")], axis=1).fillna(0.0).reset_index()
    j["error"] = j.actual - j.plan
    return j[(j.actual > 0) | (j.plan > 0)]


def policy(items, plan_weekly, deliveries, led, data_date, req=None, service=0.95, capital_rate=CAPITAL_RATE):
    """EOQ, safety stock and reorder point per bulk item at network level.

    Demand here is schedule-driven (dependent demand), so its planned ups and downs are known and handled by MRP.
    Safety stock only covers what is NOT known: the forecast error (actual issues vs plan) and lead-time variability:
        SS = z * sqrt(L * sd_err^2 + d^2 * sd_L^2)      (weekly units, L in weeks)
    Equipment is engineer-to-order: tracked against the schedule's need dates, not a reorder point."""
    z = float(stats.norm.ppf(service))
    ls = lead_stats(items, deliveries)
    err = forecast_errors(led, plan_weekly, data_date)
    net_err = err.groupby(["item_id", "week"])[["actual", "plan", "error"]].sum().reset_index()
    pw = plan_weekly.groupby(["item_id", "week"]).planned_qty.sum().reset_index()
    fut = req.groupby(["item_id", "week"]).gross.sum().reset_index() if req is not None and len(req) else None
    rows = []
    for it in items.itertuples():
        s = pw[(pw.item_id == it.item_id) & (pw.planned_qty > 0)].planned_qty
        if not len(s):
            continue
        f = fut[(fut.item_id == it.item_id) & (fut.gross > 0)].gross if fut is not None else s
        d_wk = float(f.mean()) if len(f) else float(s.mean())
        e = net_err[net_err.item_id == it.item_id]
        sd_e = float(e.error.std(ddof=1)) if len(e) > 2 else 0.25 * d_wk
        mean_act = float(e.actual[e.actual > 0].mean()) if (e.actual > 0).any() else d_wk
        act_w = e.actual[e.actual > 0]
        cv_dem = float(act_w.std(ddof=1) / act_w.mean()) if len(act_w) > 2 else np.nan
        L, sL = float(ls.lead_mean[it.item_id]) / 7, float(ls.lead_sd[it.item_id]) / 7
        annual = float(s.sum()) / max(1.0, (pw.week.max() - pw.week.min()).days / 365)
        H = it.unit_cost_usd * (capital_rate + it.storage_rate)
        S_ = ORDER_COST.get(it.origin, IMPORT_ORDER_COST)
        eoq = float(np.sqrt(2 * annual * S_ / H)) if H > 0 else np.nan
        ss = z * np.sqrt(L * sd_e ** 2 + (d_wk * sL) ** 2)
        eto = it.family == "Equipment"
        rows.append({"item_id": it.item_id, "item": it.name, "family": it.family, "unit": it.unit, "origin": it.origin,
                     "weekly_demand": d_wk, "sd_forecast_error_wk": sd_e, "lead_mean": L * 7, "lead_sd": sL * 7,
                     "annual_usage_value": annual * it.unit_cost_usd, "cv_error": sd_e / mean_act if mean_act else np.nan, "cv_demand": cv_dem, "eoq": eoq,
                     "order_qty": np.nan if eto else float(max(it.lot_size, np.ceil(eoq / it.lot_size) * it.lot_size)),
                     "safety_stock": np.nan if eto else ss, "reorder_point": np.nan if eto else d_wk * L + ss,
                     "policy": "Engineer-to-order: track against need dates" if eto else
                               ("Blanket order, call-offs to site" if it.flow == "direct" else "Reorder point / EOQ at network level"),
                     "holding_cost_unit_yr": H, "order_cost": S_})
    p = pd.DataFrame(rows)
    v = p.sort_values("annual_usage_value", ascending=False)
    cum = v.annual_usage_value.cumsum() / v.annual_usage_value.sum()
    abc = pd.Series(np.where(cum <= 0.80, "A", np.where(cum <= 0.95, "B", "C")), index=v.index)
    abc.iloc[0] = "A"
    p["abc"] = abc
    # XYZ on the variability of actual weekly consumption (coefficient of variation over active weeks)
    p["xyz"] = np.where(p.cv_demand < 0.5, "X", np.where(p.cv_demand < 1.0, "Y", "Z"))
    return p


def pooling(items, plan_weekly, deliveries, led, data_date, service=0.95):
    """Safety stock if every active site keeps its own buffer vs one pooled buffer at the hub.

    Forecast errors at different sites partly cancel, so a pooled buffer needs sqrt(sum of variances), not the sum of
    standard deviations. Computed week by week over the sites that are actually consuming the item that week."""
    z = float(stats.norm.ppf(service))
    ls = lead_stats(items, deliveries)
    err = forecast_errors(led, plan_weekly, data_date)
    rows = []
    for it in items[(items.family != "Equipment") & (items.flow == "hub")].itertuples():
        sd = err[err.item_id == it.item_id].groupby("project_id").error.std(ddof=1).dropna()
        if len(sd) < 2:
            continue
        L = float(ls.lead_mean[it.item_id]) / 7
        act = plan_weekly[(plan_weekly.item_id == it.item_id) & (plan_weekly.planned_qty > 0)].groupby("week").project_id.apply(set)
        sep, poo = [], []
        for sites in act:
            s_ = sd.reindex(list(sites)).dropna()
            if len(s_):
                sep.append(float((z * s_ * np.sqrt(L)).sum()))
                poo.append(float(z * np.sqrt((s_ ** 2).sum()) * np.sqrt(L)))
        if not sep:
            continue
        separate, pooled = float(np.mean(sep)), float(np.mean(poo))
        rows.append({"item_id": it.item_id, "item": it.name, "unit": it.unit, "sites": len(sd), "avg_sites_active": float(act.map(len).mean()),
                     "ss_separate": separate, "ss_pooled": pooled, "saving_units": separate - pooled,
                     "saving_usd": (separate - pooled) * it.unit_cost_usd, "saving_%": 100 * (1 - pooled / separate) if separate else np.nan})
    return pd.DataFrame(rows)


def equipment_tracker(items, req, bal, open_pos, windows, data_date):
    """Long-lead equipment: for each building, is it on site, and will what is on order arrive before the task needs it?"""
    eq = items[items.family == "Equipment"].set_index("item_id")
    w = windows.set_index(["project_id", "task_id"])
    rows = []
    for (p, item), g in req[req.item_id.isin(eq.index)].groupby(["project_id", "item_id"]):
        need_qty = g.gross.sum()
        need_date = pd.Timestamp(w.start.get((p, eq.task_id[item]), g.week.min()))
        on_site = float(bal[(bal.location_id == f"SITE-{p}") & (bal.item_id == item)].on_hand.sum())
        po = open_pos[(open_pos.item_id == item) & (open_pos.project_id == p)].drop_duplicates("po_id")
        on_order, due = float(po.qty.sum()), po.due_date.max() if len(po) else pd.NaT
        gap = need_qty - on_site - on_order
        slack = (need_date - due).days if pd.notna(due) else np.nan
        status = ("On site" if on_site >= need_qty - 1e-6 else
                  "NOT ORDERED: shortfall" if gap > 1e-6 else
                  "LATE vs need date" if slack < 0 else
                  "At risk (under 2 weeks' margin)" if slack < 14 else "On track")
        rows.append({"project_id": p, "item_id": item, "item": eq.name[item], "origin": eq.origin[item], "needed": need_qty, "on_site": on_site,
                     "on_order": on_order, "delivery_due": due, "need_date": need_date, "margin_days": slack, "status": status,
                     "value_usd": need_qty * eq.unit_cost_usd[item]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- transfers (transshipment LP)
def miles(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h)) * 1.25   # road factor


def transfer_plan(item, bal, req, locations, data_date, days=14, expedite_cost_per_unit=None, unit_cost=0.0):
    """Min-cost moves from stores with surplus to sites with a shortage over the next `days`. A dummy 'new purchase'
    source at a high cost covers any network shortage, so the LP is always feasible and the gap is visible."""
    end = data_date + pd.Timedelta(days=days)
    need = req[(req.item_id == item) & (req.week <= end)].groupby("project_id").gross.sum()
    loc = locations.set_index("location_id")
    oh = bal[bal.item_id == item].set_index("location_id").on_hand
    demand, supply = {}, {}
    for l in loc.index:
        have = float(oh.get(l, 0.0))
        p = loc.project_id.get(l, "")
        is_site = loc.tier.get(l) == "site"
        want = float(need.get(p, 0.0)) if is_site else 0.0
        if have < 0 and not is_site:
            continue   # negative stock at a store is a data error (see Data checks), not a demand
        if is_site and want - have > 1e-6:
            demand[l] = want - have
        elif have - want > 1e-6:
            supply[l] = have - want
    if not demand:
        return None
    per_truck = UNITS_PER_TRUCK.get(item, 1000)
    exp = expedite_cost_per_unit if expedite_cost_per_unit is not None else max(5 * TRUCK_COST_PER_MILE * 300 / per_truck, 0.15 * unit_cost)
    src = list(supply) + ["NEW PURCHASE"]
    dst = list(demand)
    cap = [supply[s] for s in supply] + [sum(demand.values())]
    cost = np.zeros((len(src), len(dst)))
    for i, s in enumerate(src):
        for j, d in enumerate(dst):
            cost[i, j] = exp if s == "NEW PURCHASE" else miles((loc.lat[s], loc.lon[s]), (loc.lat[d], loc.lon[d])) * TRUCK_COST_PER_MILE / per_truck
    n, m = cost.shape
    A_ub = np.zeros((n, n * m))
    for i in range(n):
        A_ub[i, i * m:(i + 1) * m] = 1
    A_eq = np.zeros((m, n * m))
    for j in range(m):
        A_eq[j, j::m] = 1
    res = linprog(cost.ravel(), A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=[demand[d] for d in dst], bounds=(0, None), method="highs")
    x = res.x.reshape(n, m)
    moves = [{"from": src[i], "to": dst[j], "qty": x[i, j], "cost_per_unit": cost[i, j], "cost": x[i, j] * cost[i, j],
              "miles": np.nan if src[i] == "NEW PURCHASE" else miles((loc.lat[src[i]], loc.lon[src[i]]), (loc.lat[dst[j]], loc.lon[dst[j]]))}
             for i in range(n) for j in range(m) if x[i, j] > 1e-6]
    duals = pd.DataFrame({"site": dst, "shortfall": [demand[d] for d in dst], "value_of_one_more_unit": res.eqlin.marginals})
    surplus_dual = pd.DataFrame({"source": src, "available": cap, "value_of_one_more_unit": -res.ineqlin.marginals})
    return {"moves": pd.DataFrame(moves), "total_cost": float(res.fun), "site_duals": duals, "source_duals": surplus_dual,
            "per_truck": per_truck, "expedite_cost": exp}


# ---------------------------------------------------------------- cash
def cash_view(bal, req, items, data_date, capital_rate=CAPITAL_RATE, cover_weeks=8):
    it = items.set_index("item_id")
    nxt = req[req.week <= data_date + pd.Timedelta(weeks=cover_weeks)].groupby("item_id").gross.sum() / cover_weeks
    b = bal.copy()
    b["storage_rate"] = b.item_id.map(it.storage_rate)
    b["carry_per_day"] = b.value_usd * (capital_rate + b.storage_rate) / 365
    by_item = b.groupby("item_id").agg(on_hand=("on_hand", "sum"), value_usd=("value_usd", "sum"), carry_per_day=("carry_per_day", "sum")).reset_index()
    by_item["weekly_need"] = by_item.item_id.map(nxt).fillna(0.0)
    by_item["weeks_of_supply"] = np.where(by_item.weekly_need > 0, by_item.on_hand / by_item.weekly_need, np.inf)
    by_item["family"] = by_item.item_id.map(it.family)
    by_item["excess_units"] = np.where(by_item.family == "Equipment", 0.0, np.maximum(0.0, by_item.on_hand - by_item.weekly_need * cover_weeks))
    by_item["excess_value"] = by_item.excess_units * by_item.item_id.map(it.unit_cost_usd)
    by_item["item"] = by_item.item_id.map(it.name)
    by_item["unit"] = by_item.item_id.map(it.unit)
    return b, by_item


def buy_timing(value, days_early, capital_rate, storage_rate, expected_escalation_annual, tariff_prob=0.0, tariff_rate=0.0,
               obsolescence_annual=0.0):
    """Cost of buying `days_early` before need vs expected saving from avoiding price rises (and a possible tariff)."""
    t = days_early / 365
    carry = value * (capital_rate + storage_rate + obsolescence_annual) * t
    esc = value * ((1 + expected_escalation_annual) ** t - 1)
    tariff = value * tariff_prob * tariff_rate
    return {"carrying_cost": carry, "avoided_escalation": esc, "avoided_tariff_expected": tariff, "net_benefit_of_buying_early": esc + tariff - carry,
            "break_even_escalation": (1 + (capital_rate + storage_rate + obsolescence_annual) * t - tariff_prob * tariff_rate) ** (1 / t) - 1 if t > 0 else np.nan}


# ---------------------------------------------------------------- suppliers and forecast accuracy
def otif(deliveries, suppliers, grace_days=2):
    d = deliveries.copy()
    d["late_days"] = (d.delivered_date - d.promised_date).dt.days
    d["on_time"] = d.late_days <= grace_days
    d["in_full"] = d.qty_delivered >= 0.98 * d.qty_ordered
    d["otif"] = d.on_time & d.in_full
    d["lead_actual"] = (d.delivered_date - d.order_date).dt.days
    g = d.groupby("supplier_id").agg(deliveries=("po_id", "count"), on_time=("on_time", "mean"), in_full=("in_full", "mean"), otif=("otif", "mean"),
                                     avg_late_days=("late_days", "mean"), p90_late_days=("late_days", lambda x: np.percentile(x, 90)),
                                     sd_late_days=("late_days", "std")).reset_index()
    return g.merge(suppliers, on="supplier_id", how="left"), d


def forecast_accuracy(led, plan_weekly, items, data_date):
    a = led[(led.txn_type == "ISSUE") & led.unit_ok & (led.date <= data_date)].copy()
    a["week"] = a.date.dt.to_period("W-SUN").dt.start_time
    act = a.groupby(["item_id", "week"]).qty_base.sum()
    pl = plan_weekly[plan_weekly.week <= data_date].groupby(["item_id", "week"]).planned_qty.sum()
    j = pd.concat([act.rename("actual"), pl.rename("plan")], axis=1).fillna(0.0).reset_index()
    rows = []
    for item, g in j.groupby("item_id"):
        g = g.sort_values("week")
        A, F = g.actual.values, g.plan.values
        if A.sum() <= 0:
            continue
        naive = np.abs(np.diff(A)).mean() if len(A) > 1 else np.nan
        mae = np.abs(A - F).mean()
        rows.append({"item_id": item, "item": items.set_index("item_id").name.get(item, item), "weeks": len(g), "actual_total": A.sum(),
                     "plan_total": F.sum(), "wape_%": 100 * np.abs(A - F).sum() / A.sum(), "bias_%": 100 * (F - A).sum() / A.sum(),
                     "mase": mae / naive if naive and naive > 0 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- data checks
def data_checks(items, led, locations, data_date):
    out = []
    key = items.groupby(["family", "unit", "unit_cost_usd", "supplier_id"]).item_id.apply(list)
    for ids in key:
        if len(ids) > 1:
            out.append(("Duplicate item in master", "High", ", ".join(ids), "Same family, unit, cost and supplier under two codes: merge them so stock is not split."))
    bad = led[~led.unit_ok]
    out.append(("Unit not defined for the item", "High" if len(bad) else "Pass", f"{len(bad)} rows",
                "Add the unit conversion to the item master or correct the entry."))
    iss = led[(led.txn_type == "ISSUE") & led.unit_ok]
    med = iss.groupby(["item_id", "project_id"]).qty_base.transform("median")
    out_rows = iss[iss.qty_base > 6 * med]
    out.append(("Quantity far above normal (likely unit error, e.g. bags keyed as tonnes)", "High" if len(out_rows) else "Pass",
                ", ".join(out_rows.txn_id.head(6)) or "none", "Check the source document; a 20x jump on cement is bags entered as tonnes."))
    dupe = led[led.duplicated("txn_id", keep=False)]
    out.append(("Duplicate transaction ID", "High" if len(dupe) else "Pass", ", ".join(dupe.txn_id.unique()[:6]) or "none",
                "The same movement was posted twice: stock is overstated or understated by that amount."))
    m = movements(led[led.unit_ok], set(locations.location_id))
    m["run"] = m.groupby(["location_id", "item_id"]).qty_base.cumsum()
    neg = m[m.run < -1e-6].drop_duplicates(["location_id", "item_id"])
    out.append(("Stock went negative", "High" if len(neg) else "Pass",
                "; ".join(f"{r.item_id} at {r.location_id} on {r.date:%d %b}" for r in neg.head(5).itertuples()) or "none",
                "A transfer or issue was posted before the receipt that feeds it. Post receipts first; block negative stock at entry."))
    stale = items[(data_date - pd.to_datetime(items.lead_time_reviewed)).dt.days > 180]
    out.append(("Lead time not reviewed in 6 months", "Medium" if len(stale) else "Pass", ", ".join(stale.item_id) or "none",
                "Planning uses the master lead time; refresh it from recent deliveries (see Suppliers)."))
    rec = led[(led.txn_type == "RECEIPT") & (led.ref == "")]
    out.append(("Receipt without a purchase order", "High" if len(rec) else "Pass", f"{len(rec)} rows", "Every receipt must reference a PO (three-way match)."))
    return pd.DataFrame(out, columns=["check", "severity", "found", "what to do"])


# ---------------------------------------------------------------- company inventory position and forecast
TRADE_STAGE = {"Civil": "Civil & substructure", "Shell": "Building shell", "Electrical": "Electrical", "Mechanical": "Mechanical",
               "Commissioning": "Commissioning", "Milestone": "Milestone"}
SITE_TRANSFER_DAYS = 3.0     # sub-store -> site replenishment lead time


def in_transit(shipments, lanes):
    """Open orders already shipped from the factory (past the manufacturing stage) count as in transit."""
    first = lanes.sort_values("seq").groupby("item_id").stage.first()
    sh = shipments.copy()
    sh["shipped"] = (sh.current_stage != sh.item_id.map(first)) & (sh.current_stage != "Not yet released to factory")
    return sh


def stock_forecast(items, req, bal, open_pos, pol, data_date, horizon_weeks=26):
    """Projected on hand per item and week (all stores), with open orders and the MRP's planned receipts for bulk items."""
    weeks = pd.date_range(data_date + pd.Timedelta(days=1), periods=horizon_weeks, freq="W-MON")
    m = mrp(items, req, bal, open_pos, pol, data_date, horizon_weeks)
    rows = [m[["item_id", "week", "projected_on_hand", "gross", "planned_receipt"]]]
    for it in items[items.family == "Equipment"].itertuples():
        g = req[req.item_id == it.item_id].groupby("week").gross.sum().reindex(weeks, fill_value=0.0)
        sr = open_pos[open_pos.item_id == it.item_id].drop_duplicates("po_id")
        sr = sr.assign(week=sr.due_date.dt.to_period("W-SUN").dt.start_time).groupby("week").qty.sum()
        early = sr[sr.index < weeks[0]].sum()
        sr = sr.reindex(weeks, fill_value=0.0)
        sr.iloc[0] += early
        pab = float(bal[bal.item_id == it.item_id].on_hand.sum()) + np.cumsum(sr.values - g.values)
        rows.append(pd.DataFrame({"item_id": it.item_id, "week": weeks, "projected_on_hand": pab, "gross": g.values, "planned_receipt": 0.0}))
    f = pd.concat(rows, ignore_index=True)
    f["value_usd"] = f.projected_on_hand.clip(lower=0) * f.item_id.map(items.set_index("item_id").unit_cost_usd)
    return f


def value_history(led, locations, items, data_date, freq="W-MON"):
    m = movements(led[led.unit_ok], set(locations.location_id))
    m = m.merge(locations[["location_id", "tier"]], on="location_id")
    m["usd"] = m.qty_base * m.item_id.map(items.set_index("item_id").unit_cost_usd)
    m = m[m.date <= data_date]
    w = m.set_index("date").groupby("tier").usd.resample(freq).sum().reset_index()
    w["value_usd"] = w.groupby("tier").usd.cumsum()
    return w


def inventory_position(items, bal, req, open_pos, shipments, lanes, fc, data_date):
    """Company-wide position per item: on hand by tier, in transit, on order (not shipped), allocated to remaining
    project need, free stock, weeks of supply and the first week the forecast runs out."""
    sh = in_transit(shipments, lanes)
    it = items.set_index("item_id")
    need = req.groupby("item_id").gross.sum()
    nxt8 = req[req.week <= data_date + pd.Timedelta(weeks=8)].groupby("item_id").gross.sum() / 8
    rows = []
    for i in items.item_id:
        b = bal[bal.item_id == i]
        transit = float(sh[(sh.item_id == i) & sh.shipped].qty.sum())
        ordered = float(sh[(sh.item_id == i) & ~sh.shipped].qty.sum())
        oh = float(b.on_hand.sum())
        f = fc[fc.item_id == i].sort_values("week")
        out = f[f.projected_on_hand < -1e-6]
        rows.append({"item_id": i, "item": it.name[i], "family": it.family[i], "unit": it.unit[i],
                     "hub": float(b[b.tier == "hub"].on_hand.sum()), "sub_stores": float(b[b.tier == "sub-store"].on_hand.sum()),
                     "sites": float(b[b.tier == "site"].on_hand.sum()), "on_hand": oh, "in_transit": transit, "on_order": ordered,
                     "allocated_to_projects": float(need.get(i, 0.0)), "free": oh + transit + ordered - float(need.get(i, 0.0)),
                     "value_on_hand": oh * it.unit_cost_usd[i], "weeks_of_supply": oh / nxt8[i] if nxt8.get(i, 0) > 0 else np.inf,
                     "forecast_runs_out": out.week.iloc[0] if len(out) else pd.NaT})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one building
def project_view(pid, items, bal, req, led, plan_weekly, data_date, service=0.95, capital_rate=CAPITAL_RATE, templates=None):
    """Everything about one building: site stock, its share of shared stock (by share of remaining need), need by part
    of the building, site safety stock and reorder point (its own forecast error, sub-store -> site lead time), cash."""
    z = float(stats.norm.ppf(service))
    it = items.set_index("item_id")
    r = req[req.project_id == pid]
    site = bal[bal.location_id == f"SITE-{pid}"].set_index("item_id").on_hand
    shared = bal[bal.tier != "site"].groupby("item_id").on_hand.sum()
    tot_need = req.groupby("item_id").gross.sum()
    my_need = r.groupby("item_id").gross.sum()
    err = forecast_errors(led, plan_weekly, data_date)
    err = err[err.project_id == pid]
    trade = dict(zip(templates.id, templates.trade)) if templates is not None else {}
    rows = []
    for i in items.item_id:
        need = float(my_need.get(i, 0.0))
        at_site = float(site.get(i, 0.0))
        if need <= 0 and at_site <= 0:
            continue
        share = need / tot_need[i] if tot_need.get(i, 0) > 0 else 0.0
        need14 = float(r[(r.item_id == i) & (r.week <= data_date + pd.Timedelta(days=14))].gross.sum())
        wk = r[r.item_id == i].groupby("week").gross.sum()
        d = float(wk[wk > 0].mean()) if (wk > 0).any() else 0.0
        e = err[err.item_id == i].error
        sd = float(e.std(ddof=1)) if len(e) > 2 else 0.25 * d
        L = SITE_TRANSFER_DAYS / 7
        eto = it.family[i] == "Equipment"
        ss = np.nan if (eto or need <= 0) else z * sd * np.sqrt(L)
        rop = np.nan if (eto or need <= 0) else d * L + ss
        rows.append({"item_id": i, "item": it.name[i], "unit": it.unit[i], "part_of_building": TRADE_STAGE.get(trade.get(it.task_id[i], ""), "Other"),
                     "task_id": it.task_id[i], "remaining_need": need, "at_site": at_site, "share_of_shared_stock": share * max(0.0, float(shared.get(i, 0.0))),
                     "covered_by_site_%": 100 * min(1.0, at_site / need) if need > 0 else 100.0, "weekly_need": d,
                     "site_safety_stock": ss, "site_reorder_point": rop,
                     "need_next_2wk": need14,
                     "call_off_now": bool((not eto) and need14 > 0 and at_site - need14 < rop) if rop == rop else False,
                     "excess_at_site": max(0.0, at_site - need), "excess_value": max(0.0, at_site - need) * it.unit_cost_usd[i],
                     "site_value": at_site * it.unit_cost_usd[i], "remaining_need_value": need * it.unit_cost_usd[i],
                     "carry_per_day": at_site * it.unit_cost_usd[i] * (capital_rate + it.storage_rate[i]) / 365,
                     "weeks_of_supply": at_site / d if d > 0 else np.inf})
    return pd.DataFrame(rows)


def site_plan(pid, item, req, bal, rop, ss, call_off_qty, data_date, horizon_weeks=16):
    """Weekly projected site stock. A call-off from the sub-store is triggered when work needs the item within two weeks
    and stock after this week would fall below the site reorder point."""
    weeks = pd.date_range(data_date + pd.Timedelta(days=1), periods=horizon_weeks, freq="W-MON")
    g = req[(req.project_id == pid) & (req.item_id == item)].groupby("week").gross.sum().reindex(weeks, fill_value=0.0)
    oh = float(bal[(bal.location_id == f"SITE-{pid}") & (bal.item_id == item)].on_hand.sum())
    rows = []
    for j, wk in enumerate(weeks):
        nxt2 = float(g.iloc[j:j + 2].sum())
        left = float(g.iloc[j:].sum())
        call = 0.0
        if nxt2 > 0 and oh - g.iloc[j] < rop:
            call = max(call_off_qty, rop - (oh - g.iloc[j]))
            keep = ss if float(g.iloc[j + 2:].sum()) > 0 else 0.0      # no buffer needed once the work is nearly done
            call = min(call, max(0.0, left + keep - oh))
        oh = oh + call - g.iloc[j]
        rows.append({"week": wk, "need": g.iloc[j], "call_off": call, "site_stock": oh})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- lead time through the supply chain
def lead_breakdown(item, lanes):
    """Stage-by-stage lead time: mean, spread, and each stage's share of the total uncertainty (variance)."""
    x = lanes[lanes.item_id == item].sort_values("seq").copy()
    x["var"] = x.sd_days ** 2
    x["share_of_uncertainty_%"] = 100 * x["var"] / x["var"].sum()
    mean, sd = x.mean_days.sum(), float(np.sqrt(x["var"].sum()))
    return x, {"mean": mean, "sd": sd, "p90": mean + 1.2816 * sd}


def _gamma(m, sd, size, rng):
    m = max(m, 1e-6)
    if sd <= 0:
        return np.full(size, m)
    k = (m / sd) ** 2
    return rng.gamma(k, m / k, size)


def pipeline(shipments, lanes, need_dates, data_date, n=4000, seed=5):
    """Where each open order is, and when it will arrive (Monte Carlo over the remaining stages), vs the date it is needed.
    need_dates: dict (po_id) -> need date (from the schedule), may be missing."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in shipments.itertuples():
        segs = lanes[lanes.item_id == r.item_id].sort_values("seq")
        stages = list(segs.stage)
        if r.current_stage in stages:
            k = stages.index(r.current_stage)
            cur = segs.iloc[k]
            rem_cur = max(0.5, cur.mean_days - r.days_in_stage)
            # remaining time in the current stage: conditional on time already spent, approximated by the stage's mean residual
            tot = _gamma(rem_cur, cur.sd_days * min(1.0, rem_cur / max(cur.mean_days, 1e-6)), n, rng)
            after = segs.iloc[k + 1:]
        else:   # not yet released: the factory starts when the order is released (promised date minus the full lead time)
            start = max(0.0, (pd.Timestamp(r.promised_due) - data_date).days - float(segs.mean_days.sum()))
            k, tot, after = -1, np.full(n, start), segs
        for sg in after.itertuples():
            tot = tot + _gamma(sg.mean_days, sg.sd_days, n, rng)
        p50, p90 = np.percentile(tot, [50, 90])
        eta50 = data_date + pd.Timedelta(days=float(p50))
        eta90 = data_date + pd.Timedelta(days=float(p90))
        need = need_dates.get(r.po_id, pd.NaT)
        p_late = float(((data_date + pd.to_timedelta(tot, unit="D")) > need).mean()) if pd.notna(need) else np.nan
        rows.append({"po_id": r.po_id, "item_id": r.item_id, "project_id": r.project_id, "qty": r.qty, "current_stage": r.current_stage,
                     "stage_no": f"{k + 1} of {len(stages)}" if k >= 0 else f"0 of {len(stages)}", "days_in_stage": r.days_in_stage,
                     "promised_due": r.promised_due, "eta_p50": eta50, "eta_p90": eta90, "need_date": need,
                     "margin_p50_days": (need - eta50).days if pd.notna(need) else np.nan, "chance_late_%": 100 * p_late if p_late == p_late else np.nan,
                     "past_promise": pd.Timestamp(r.promised_due) < data_date})
    out = pd.DataFrame(rows)
    out["flag"] = np.where(out["chance_late_%"] >= 50, "Likely late vs need", np.where(out["chance_late_%"] >= 20, "At risk",
                           np.where(out.past_promise, "Past promised date", "OK")))
    return out


def need_dates_for_orders(shipments, equipment, mrp_table, open_pos, data_date):
    """Need date per open order: equipment -> the building's task start; bulk replenishment -> the first week the network
    would drop below safety stock without that order (approximation: orders in due-date order cover successive shortfalls)."""
    nd = {}
    for r in shipments.itertuples():
        if isinstance(r.project_id, str) and r.project_id:
            m = equipment[(equipment.project_id == r.project_id) & (equipment.item_id == r.item_id)]
            if len(m):
                nd[r.po_id] = m.need_date.iloc[0]
    for item, g in shipments[shipments.po_id.map(lambda x: x not in nd)].groupby("item_id"):
        m = mrp_table[mrp_table.item_id == item].sort_values("week")
        if not len(m):
            continue
        sr_cum = (m.scheduled_receipts.cumsum())
        pab_wo = m.projected_on_hand - m.planned_receipt.cumsum() - sr_cum     # stock without any open or planned order
        short_weeks = m.week[pab_wo.values < m.safety_stock.values]
        orders = g.sort_values("promised_due")
        need_cum = (m.safety_stock - pab_wo).clip(lower=0)
        q = 0.0
        for o in orders.itertuples():
            q += o.qty
            hit = m.week[need_cum.values > q - o.qty + 1e-6]
            nd[o.po_id] = hit.iloc[0] if len(hit) else pd.NaT
    return nd
