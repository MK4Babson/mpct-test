"""Materials & supply chain: one stock ledger across hub, sub-stores and sites; MRP from the CPM schedule; six-week
lookahead; inventory policy (ABC-XYZ, EOQ, safety stock, pooling); transfers (LP); cash tied up; suppliers; data checks."""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pydeck as pdk

import inventory as I
import schedule as S
from insights import money
from ui.cost_pages import back, hero, howto, synthetic_note
from ui.theme import SERIES, STATUS

ROOT = Path(__file__).resolve().parents[1]
DOLLAR = st.column_config.NumberColumn(format="$%,.0f")
NUM = st.column_config.NumberColumn(format="%,.1f")
DATA_DATE = pd.Timestamp("2026-08-31")
TIER_COLOR = alt.Scale(domain=["hub", "sub-store", "site"], range=[SERIES[0], SERIES[1], SERIES[2]])


@st.cache_data(show_spinner=False)
def _base():
    return I.load()


def ledger():
    if "scm_ledger" not in st.session_state:
        st.session_state["scm_ledger"] = _base()["ledger"].copy()
    return st.session_state["scm_ledger"]


def _projects():
    return pd.read_csv(ROOT / "data" / "raw" / "projects.csv")


def windows(ctx):
    """Task windows from the live CPM forecast when the schedule is available for these buildings; else the baseline."""
    from ui import schedule_page as SP
    d = _base()
    rows, src = [], "CPM forecast (Schedule page)"
    for pid in _projects().project_id:
        try:
            t, meta = SP.network(ctx, pid)
        except Exception:
            t = None
        if t is None:
            src = "baseline schedule (no live schedule for these buildings in the selected data)"
            rows = []
            break
        f = S.forecast(t, meta)
        st_ = S.dates(meta, f.fstart.fillna(f.bl_finish if "bl_finish" in f else f.ef))
        fi_ = S.dates(meta, f.ffinish.fillna(f.bl_finish if "bl_finish" in f else f.ef))
        for a, s_, e_ in zip(f.id, st_, fi_):
            rows.append({"project_id": pid, "task_id": a, "start": s_, "finish": e_})
    if not rows:
        w = d["baseline_windows"].rename(columns={"bl_start": "start", "bl_finish": "finish"})
        return w, src
    return pd.DataFrame(rows), src


def model(ctx, service=0.95, capital_rate=I.CAPITAL_RATE):
    """Everything the materials views need. Memoised per session on the ledger and settings, so posting one movement
    recomputes every view once."""
    led_ = ledger()
    key = (len(led_), str(led_.txn_id.iloc[-1]), float(led_.qty.iloc[-1]), service, capital_rate, getattr(ctx, "source_key", ""), str(getattr(ctx, "end_period", "")))
    memo = st.session_state.setdefault("scm_memo", {})
    if key not in memo:
        memo.clear()
        memo[key] = _model(ctx, service, capital_rate)
    return memo[key]


def _model(ctx, service, capital_rate):
    d = _base()
    items = d["items"]
    plan_items = items[items.item_id != "CEM-OPC-B"] if "CEM-OPC-B" in set(items.item_id) else items
    led = I.to_base(ledger(), items)
    led_clean = I.clean(led)
    w, wsrc = windows(ctx)
    bal = I.balances(led_clean, d["locations"], items, DATA_DATE)
    req = I.requirements(plan_items, _projects(), w, I.issued(led_clean, DATA_DATE), DATA_DATE)
    pol = I.policy(plan_items, d["plan_weekly"], d["deliveries"], led_clean, DATA_DATE, req, service, capital_rate)
    m = I.mrp(plan_items, req, bal, d["open_pos"], pol, DATA_DATE)
    eq = I.equipment_tracker(plan_items, req, bal, d["open_pos"], w, DATA_DATE)
    fc = I.stock_forecast(plan_items, req, bal, d["open_pos"], pol, DATA_DATE)
    nd = I.need_dates_for_orders(d["shipments"], eq, m, d["open_pos"], DATA_DATE)
    pipe = I.pipeline(d["shipments"], d["lanes"], nd, DATA_DATE)
    pos = I.inventory_position(plan_items, bal, req, d["open_pos"], d["shipments"], d["lanes"], fc, DATA_DATE)
    return {"d": d, "items": items, "plan_items": plan_items, "led": led, "led_clean": led_clean, "windows": w, "wsrc": wsrc, "bal": bal,
            "req": req, "pol": pol, "mrp": m, "eq": eq, "fc": fc, "pipe": pipe, "pos": pos}


def _fmt_qty(q, unit):
    return f"{q:,.0f} {unit}" if abs(q) >= 10 else f"{q:,.1f} {unit}"


# ---------------------------------------------------------------- page
def page(ctx):
    back(ctx, "scm")
    hero("Materials & supply chain", "Every bag, reel and transformer, everywhere, against the schedule",
         "One stock ledger feeds every view: a single movement at the hub, a sub-store or a site updates balances, the material "
         "plan, the six-week lookahead, cash tied up and the data checks. Demand comes from the CPM schedule.")
    synthetic_note("Synthetic materials data (items, suppliers, ledger, deliveries) for the six West Texas buildings. "
                   "Item master, ledger and open orders can be replaced with real exports in the same columns.")
    with st.expander("Settings", expanded=False):
        c1, c2 = st.columns(2)
        service = c1.slider("Service level for safety stock", 0.80, 0.995, 0.95, 0.005,
                            help="Probability of not running out during a replenishment lead time. 95% -> z = 1.645.")
        cap = c2.slider("Owner's cost of capital (carrying cost)", 0.03, 0.15, I.CAPITAL_RATE, 0.005, format="%.3f")
    M = model(ctx, service, cap)
    d, bal, req, pol, items = M["d"], M["bal"], M["req"], M["pol"], M["items"]

    k = st.columns(5)
    k[0].metric("Stock value, all stores", money(bal.value_usd.sum()))
    k[1].metric("At hub + sub-stores", money(bal[bal.tier != "site"].value_usd.sum()))
    k[2].metric("At site stores", money(bal[bal.tier == "site"].value_usd.sum()))
    _, ci = I.cash_view(bal, req, items, DATA_DATE, cap)
    k[3].metric("Carrying cost per day", money(ci.carry_per_day.sum()), help="Value x (cost of capital + storage) / 365")
    la = I.lookahead(M["plan_items"], req, bal, d["open_pos"], M["windows"], d["locations"], DATA_DATE)
    k[4].metric("6-week constraints", int((la.status == "CONSTRAINT").sum()), help="Tasks starting in the next 6 weeks whose material is not secured")
    st.caption(f"Data date {DATA_DATE:%d %b %Y} · task dates from the {M['wsrc']}.")

    tabs = st.tabs(["Company inventory", "Project view", "Supply chain & lead times", "Stock everywhere", "Material plan (MRP)", "6-week lookahead",
                    "Inventory policy", "Transfers", "Cash & buy timing", "Suppliers & forecast", "Data checks"])
    with tabs[0]:
        _company(M)
    with tabs[1]:
        _project(M, service, cap)
    with tabs[2]:
        _network(M)
    with tabs[3]:
        _stock(M)
    with tabs[4]:
        _mrp(M)
    with tabs[5]:
        _lookahead(M, la)
    with tabs[6]:
        _policy(M, service)
    with tabs[7]:
        _transfers(M)
    with tabs[8]:
        _cash(M, ci, cap)
    with tabs[9]:
        _suppliers(M)
    with tabs[10]:
        _checks(M, service, cap)


# ---------------------------------------------------------------- tabs
def _stock(M):
    d, bal, items = M["d"], M["bal"], M["items"]
    howto("Pick an item to see where it is. Balances are the sum of every movement in the ledger. "
          "Use <b>Post a movement</b> to move stock (for example 40 bags of cement from the hub to a site) and watch every view update.")
    pi = M["plan_items"]
    lab = dict(zip(pi.item_id, pi.item_id + " · " + pi.name))
    item = st.selectbox("Item", list(lab), format_func=lab.get, key="scm_item")
    it = items.set_index("item_id").loc[item]
    b = bal[bal.item_id == item].copy()
    locs = d["locations"][["location_id", "name", "tier"]]
    b = locs.merge(b[["location_id", "on_hand", "value_usd"]], on="location_id", how="left").fillna({"on_hand": 0.0, "value_usd": 0.0})
    alt_u = it.alt_unit
    if alt_u:
        b[f"on_hand ({alt_u})"] = b.on_hand * it.alt_per_unit
    c1, c2 = st.columns([3, 2])
    ch = alt.Chart(b).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("on_hand:Q", title=f"On hand ({it.unit})"), y=alt.Y("name:N", sort=list(locs.name), title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color("tier:N", scale=TIER_COLOR, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["name", "tier", alt.Tooltip("on_hand:Q", format=",.1f"), alt.Tooltip("value_usd:Q", format="$,.0f")])
    c1.altair_chart(ch.properties(height=alt.Step(22)), width="stretch")
    with c2:
        st.metric(f"Total on hand ({it.unit})", f"{b.on_hand.sum():,.1f}")
        if alt_u:
            st.metric(f"Same, in {alt_u}", f"{b.on_hand.sum() * it.alt_per_unit:,.0f}")
            st.caption(f"Master data: 1 {it.unit} = {it.alt_per_unit:g} {alt_u}. Entries in either unit are converted to {it.unit}.")
        st.metric("Value", money(b.value_usd.sum()))
        st.caption(f"Origin {it.origin} · supplier {it.supplier_id} · master lead time {it.lead_days:.0f} days")
    with st.expander("Post a movement (demo of one ledger driving everything)", expanded=False):
        stores = list(d["locations"].location_id)
        f1, f2, f3 = st.columns(3)
        kind = f1.selectbox("Type", ["TRANSFER", "ISSUE", "RECEIPT", "ADJUST"], key="pm_kind")
        frm = f2.selectbox("From", (["SUPPLIER"] if kind == "RECEIPT" else []) + stores, key="pm_from")
        to = f3.selectbox("To", (["(consumed on site)"] if kind in ("ISSUE", "ADJUST") else []) + stores, index=0, key="pm_to")
        g1, g2, g3 = st.columns(3)
        units = [it.unit] + ([alt_u] if alt_u else [])
        unit = g1.selectbox("Unit", units, key="pm_unit")
        qty = g2.number_input("Quantity", 0.0, 1e7, 40.0, key="pm_qty")
        date = g3.date_input("Date", DATA_DATE, key="pm_date")
        if st.button("Post movement"):
            to_ = "" if to.startswith("(") else to
            frm_ = it.supplier_id if frm == "SUPPLIER" else frm
            proj = d["locations"].set_index("location_id").project_id.get(to_ or frm_, "")
            st.session_state["scm_ledger"] = I.post(ledger(), date, item, frm_, to_, qty, unit, kind, proj if isinstance(proj, str) else "")
            st.success(f"Posted {qty:g} {unit} of {item} {frm} → {to or 'consumed'}. Every tab now uses the new balance.")
            st.rerun()
        if (ledger().note == "posted in app").any():
            st.dataframe(ledger()[ledger().note == "posted in app"], hide_index=True, width="stretch")
            if st.button("Undo all posted movements"):
                st.session_state["scm_ledger"] = _base()["ledger"].copy()
                st.rerun()
    with st.expander("Movement log for this item"):
        lg = M["led"][M["led"].item_id == item].sort_values("date", ascending=False)
        st.dataframe(lg[["txn_id", "date", "txn_type", "from_loc", "to_loc", "qty", "unit", "qty_base", "project_id", "ref", "note"]].head(300),
                     hide_index=True, width="stretch", column_config={"date": st.column_config.DateColumn(format="DD MMM YYYY")})


def _mrp(M):
    d, req, pol, bal = M["d"], M["req"], M["pol"], M["bal"]
    howto("Material requirements planning, network level. <b>Gross</b> = remaining material for each building spread over its task's "
          "forecast window. <b>Projected on hand</b> = stock + open orders − gross. When it would fall below safety stock, a planned "
          "receipt appears, and its <b>release date</b> = need week − lead time. A release date already passed means expedite.")
    m = I.mrp(M["plan_items"], req, bal, d["open_pos"], pol, DATA_DATE)
    act = m[m.status != ""]
    c1, c2, c3 = st.columns(3)
    c1.metric("Expedite actions", int(act.status.str.startswith("EXPEDITE").sum()))
    c2.metric("Orders to release in 2 weeks", int((act.status == "Release order").sum()))
    c3.metric("Call-offs on blanket orders", int(act.status.str.startswith("Call-off").sum()))
    if len(act):
        st.dataframe(act.merge(M["items"][["item_id", "name", "unit"]], on="item_id")[
            ["item_id", "name", "week", "gross", "projected_on_hand", "safety_stock", "planned_receipt", "planned_release", "status", "unit"]],
            hide_index=True, width="stretch", column_config={"week": st.column_config.DateColumn("need week", format="DD MMM"),
                                                            "planned_release": st.column_config.DateColumn(format="DD MMM YYYY"),
                                                            "gross": NUM, "projected_on_hand": NUM, "safety_stock": NUM, "planned_receipt": NUM})
    items_m = sorted(m.item_id.unique())
    lab = dict(zip(M["items"].item_id, M["items"].item_id + " · " + M["items"].name))
    item = st.selectbox("Plan for item", items_m, format_func=lab.get, key="mrp_item")
    x = m[m.item_id == item]
    base = alt.Chart(x).encode(x=alt.X("week:T", title=None))
    bars = base.mark_bar(color=SERIES[1], opacity=0.75).encode(y=alt.Y("gross:Q", title="Units per week / on hand"), tooltip=["week:T", alt.Tooltip("gross:Q", format=",.0f")])
    line = base.mark_line(color=SERIES[0], strokeWidth=2, point=True).encode(y="projected_on_hand:Q", tooltip=["week:T", alt.Tooltip("projected_on_hand:Q", format=",.0f")])
    ss = base.mark_rule(color=STATUS["warning"], strokeDash=[5, 4]).encode(y="mean(safety_stock):Q")
    st.altair_chart((bars + line + ss).properties(height=260, title="Gross requirement (bars), projected on hand (line), safety stock (dashed)"),
                    width="stretch")
    st.markdown("#### Long-lead equipment against need dates")
    howto("Engineer-to-order equipment is not managed by reorder points. For each building: is it on site, and will what is on order "
          "arrive before the schedule needs it? Margin = need date − delivery due date.")
    eq = I.equipment_tracker(M["plan_items"], req, bal, d["open_pos"], M["windows"], DATA_DATE)
    if len(eq):
        st.dataframe(eq.sort_values("margin_days"), hide_index=True, width="stretch",
                     column_config={"delivery_due": st.column_config.DateColumn(format="DD MMM YYYY"), "need_date": st.column_config.DateColumn(format="DD MMM YYYY"),
                                    "value_usd": DOLLAR, "needed": NUM, "on_site": NUM, "on_order": NUM})
    else:
        st.info("No remaining equipment requirement.")


def _lookahead(M, la):
    howto("Last Planner style: work can only start when its constraints are removed. For every task starting in the next six weeks, is "
          "the material at the site, in the campus store, or arriving in time? Each constraint has a <b>Responsible</b> and an "
          "<b>Accountable</b> owner (RACI) so it gets cleared before the week starts.")
    proj = st.multiselect("Buildings", sorted(la.project_id.unique()), default=sorted(la.project_id.unique()), key="la_p")
    x = la[la.project_id.isin(proj)]
    st.dataframe(x, hide_index=True, width="stretch",
                 column_config={"task_start": st.column_config.DateColumn(format="DD MMM"), "need_6wk": NUM, "at_site": NUM, "at_campus_store": NUM,
                                "incoming_by_then": NUM, "shortfall": NUM})
    con = x[x.status == "CONSTRAINT"]
    if len(con):
        st.warning(f"{len(con)} constraints. Clear them in this week's coordination meeting: transfer from another store (see Transfers), "
                   "expedite the order, or resequence the task.")


def _policy(M, service):
    pol = M["pol"]
    howto("<b>ABC</b>: share of annual usage value (A = top 80%). <b>XYZ</b>: how steady weekly consumption is (coefficient of variation: X < 0.5, Y < 1, Z ≥ 1). "
          "Demand here is driven by the schedule, so safety stock covers only what is not known: the forecast error and lead-time "
          "variability. <b>SS = z·√(L·σ²error + d²·σ²L)</b>, <b>reorder point = d·L + SS</b>, <b>EOQ = √(2DS/H)</b>.")
    mat = pol.groupby(["abc", "xyz"]).agg(items=("item_id", "count"), value=("annual_usage_value", "sum")).reset_index()
    ch = alt.Chart(mat).mark_rect(cornerRadius=4).encode(
        x=alt.X("xyz:N", title="XYZ (variability)", sort=["X", "Y", "Z"]), y=alt.Y("abc:N", title="ABC (value)", sort=["A", "B", "C"]),
        color=alt.Color("value:Q", scale=alt.Scale(range=["#1c2a4a", SERIES[0]]), legend=None), tooltip=["abc", "xyz", "items", alt.Tooltip("value:Q", format="$,.0f")])
    txt = alt.Chart(mat).mark_text(color="#1f1f1f", fontSize=14).encode(x=alt.X("xyz:N", sort=["X", "Y", "Z"]), y=alt.Y("abc:N", sort=["A", "B", "C"]),
                                                                         text=alt.Text("items:Q"))
    c1, c2 = st.columns([1, 2])
    c1.altair_chart((ch + txt).properties(height=200, title="Items by class"), width="stretch")
    c2.markdown("- **AX / AY**: high value, predictable: tight control, frequent small call-offs, low buffer.\n"
                "- **A-class equipment**: engineer-to-order, tracked against need dates (Material plan tab).\n"
                "- **CZ**: low value, erratic: keep a simple buffer, order in bigger lots, don't spend planner time.")
    st.dataframe(pol[["item_id", "item", "abc", "xyz", "cv_demand", "policy", "unit", "weekly_demand", "sd_forecast_error_wk", "lead_mean", "lead_sd", "eoq",
                      "order_qty", "safety_stock", "reorder_point", "holding_cost_unit_yr", "order_cost"]],
                 hide_index=True, width="stretch",
                 column_config={c: NUM for c in ["weekly_demand", "sd_forecast_error_wk", "lead_mean", "lead_sd", "eoq", "order_qty", "safety_stock", "reorder_point"]}
                 | {"holding_cost_unit_yr": st.column_config.NumberColumn(format="$%,.2f"), "order_cost": DOLLAR})
    st.markdown("#### Risk pooling: one buffer at the hub vs one per site")
    howto("Forecast errors at different sites partly cancel, so one shared buffer needs √(sum of variances) instead of the sum of each "
          "site's buffer. Computed week by week over the sites that are actually consuming the item.")
    pl = I.pooling(M["plan_items"], M["d"]["plan_weekly"], M["d"]["deliveries"], M["led_clean"], DATA_DATE, service)
    st.metric("Safety-stock value saved by pooling at the hub", money(pl.saving_usd.sum()))
    st.dataframe(pl, hide_index=True, width="stretch", column_config={"ss_separate": NUM, "ss_pooled": NUM, "saving_units": NUM, "saving_usd": DOLLAR,
                                                                      "saving_%": st.column_config.NumberColumn(format="%.0f%%"), "avg_sites_active": NUM})


def _transfers(M):
    d, bal, req = M["d"], M["bal"], M["req"]
    howto("Minimum-cost transshipment (linear program, HiGHS). Sources: stores holding more than their own next-two-weeks need. "
          "Destinations: sites short for the next two weeks. A <b>NEW PURCHASE</b> source at a high cost covers any network gap, so the "
          "answer always shows how much must be bought or expedited. <b>Value of one more unit</b> is the LP's shadow price.")
    hub_items = M["plan_items"][(M["plan_items"].flow == "hub")]
    lab = dict(zip(hub_items.item_id, hub_items.item_id + " · " + hub_items.name))
    c1, c2 = st.columns([2, 1])
    item = c1.selectbox("Item", list(lab), format_func=lab.get, key="tp_item")
    days = c2.slider("Horizon (days)", 7, 42, 14, 7, key="tp_days")
    it = M["items"].set_index("item_id").loc[item]
    tp = I.transfer_plan(item, bal, req, d["locations"], DATA_DATE, days, unit_cost=it.unit_cost_usd)
    if tp is None:
        st.success("No site is short of this item in the horizon.")
        return
    mv = tp["moves"]
    a, b, c = st.columns(3)
    a.metric("Moves", len(mv[mv["from"] != "NEW PURCHASE"]))
    b.metric("Transfer cost", money(mv[mv["from"] != "NEW PURCHASE"].cost.sum()))
    c.metric(f"Must buy / expedite ({it.unit})", f"{mv[mv['from'] == 'NEW PURCHASE'].qty.sum():,.0f}")
    st.dataframe(mv, hide_index=True, width="stretch", column_config={"qty": NUM, "cost_per_unit": st.column_config.NumberColumn(format="$%.3f"),
                                                                      "cost": DOLLAR, "miles": NUM})
    st.caption(f"Truck cost ${I.TRUCK_COST_PER_MILE}/mile, {tp['per_truck']:,} {it.unit} per truck, road distance = straight line x 1.25. "
               f"New purchase / expedite priced at ${tp['expedite_cost']:,.2f} per {it.unit}.")
    c1, c2 = st.columns(2)
    c1.markdown("**Value of one more unit at each short site**")
    c1.dataframe(tp["site_duals"], hide_index=True, width="stretch", column_config={"shortfall": NUM, "value_of_one_more_unit": st.column_config.NumberColumn(format="$%.3f")})
    c2.markdown("**Value of one more unit at each source**")
    c2.dataframe(tp["source_duals"], hide_index=True, width="stretch", column_config={"available": NUM, "value_of_one_more_unit": st.column_config.NumberColumn(format="$%.3f")})


def _cash(M, ci, cap):
    howto("Money sitting in stock is money not moving. <b>Carrying cost</b> = value × (cost of capital + storage) per day. <b>Weeks of supply</b> = "
          "on hand ÷ average weekly need over the next 8 weeks. Stock beyond 8 weeks of need is <b>excess</b>: transfer it, return it, or stop the next call-off.")
    bal = M["bal"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cash tied up in excess bulk stock", money(ci.excess_value.sum()))
    c2.metric("Carrying cost of all stock, per month", money(ci.carry_per_day.sum() * 30))
    eq_wait = bal[(bal.family == "Equipment") & (bal.on_hand > 0)].value_usd.sum()
    c3.metric("Equipment delivered, waiting to be set", money(eq_wait))
    t = ci[["item_id", "item", "family", "unit", "on_hand", "value_usd", "carry_per_day", "weekly_need", "weeks_of_supply", "excess_units", "excess_value"]]
    st.dataframe(t.sort_values("excess_value", ascending=False), hide_index=True, width="stretch",
                 column_config={"value_usd": DOLLAR, "carry_per_day": st.column_config.NumberColumn(format="$%,.0f"), "excess_value": DOLLAR,
                                "on_hand": NUM, "weekly_need": NUM, "weeks_of_supply": NUM, "excess_units": NUM})
    tier = bal.groupby("tier").value_usd.sum().reset_index()
    st.altair_chart(alt.Chart(tier).mark_bar(cornerRadiusEnd=4).encode(x=alt.X("value_usd:Q", title="Stock value ($)"), y=alt.Y("tier:N", title=None, sort=["hub", "sub-store", "site"]),
                                                                       color=alt.Color("tier:N", scale=TIER_COLOR, legend=None),
                                                                       tooltip=["tier", alt.Tooltip("value_usd:Q", format="$,.0f")]).properties(height=110), width="stretch")
    st.markdown("#### Buy early or buy late?")
    howto("Buying early ties up cash and storage; buying late risks price rises and a tariff change. The calculator compares the two. "
          "Use the Global markets page for the tariff table and FX.")
    a, b, c = st.columns(3)
    value = a.number_input("Order value ($)", 10_000.0, 50_000_000.0, 1_000_000.0, 50_000.0)
    days = b.slider("Days earlier than needed", 0, 365, 90, 15)
    storage = c.slider("Storage + handling (% of value / yr)", 0.0, 0.10, 0.03, 0.005, format="%.3f")
    a, b, c = st.columns(3)
    esc = a.slider("Expected price escalation (% / yr)", -0.10, 0.25, 0.06, 0.01, format="%.2f")
    p_t = b.slider("Chance a new tariff applies before you'd buy", 0.0, 1.0, 0.0, 0.05)
    r_t = c.slider("Size of that tariff", 0.0, 0.5, 0.25, 0.05)
    r = I.buy_timing(value, days, cap, storage, esc, p_t, r_t)
    a, b, c, e = st.columns(4)
    a.metric("Carrying cost", money(r["carrying_cost"]))
    b.metric("Escalation avoided", money(r["avoided_escalation"]))
    c.metric("Expected tariff avoided", money(r["avoided_tariff_expected"]))
    e.metric("Net benefit of buying early", money(r["net_benefit_of_buying_early"]), "buy early" if r["net_benefit_of_buying_early"] > 0 else "buy late",
             delta_color="normal" if r["net_benefit_of_buying_early"] > 0 else "inverse")
    if days:
        st.caption(f"Break-even escalation: buying early pays only if prices rise faster than {100 * r['break_even_escalation']:.1f}% a year "
                   "(given the tariff risk entered).")


def _suppliers(M):
    d = M["d"]
    howto("<b>OTIF</b> = share of deliveries that were both on time (within 2 days of the promise) and in full (≥ 98% of quantity). "
          "Late-day spread is what safety stock and schedule buffers have to absorb.")
    o, det = I.otif(d["deliveries"], d["suppliers"])
    ch = alt.Chart(o).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("otif:Q", title="OTIF", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])), y=alt.Y("name:N", sort="x", title=None),
        color=alt.condition(alt.datum.otif < 0.6, alt.value(STATUS["serious"]), alt.value(SERIES[0])),
        tooltip=["name", "country", "deliveries", alt.Tooltip("on_time:Q", format=".0%"), alt.Tooltip("in_full:Q", format=".0%"), alt.Tooltip("otif:Q", format=".0%"),
                 alt.Tooltip("avg_late_days:Q", format=".1f")])
    st.altair_chart(ch.properties(height=alt.Step(22)), width="stretch")
    st.dataframe(o[["supplier_id", "name", "country", "currency", "deliveries", "on_time", "in_full", "otif", "avg_late_days", "p90_late_days", "sd_late_days"]],
                 hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format="%.0f%%") for k in []}
                 | {"on_time": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1), "in_full": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                    "otif": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1), "avg_late_days": NUM, "p90_late_days": NUM, "sd_late_days": NUM})
    ls = I.lead_stats(M["items"], d["deliveries"]).reset_index().rename(columns={"index": "item_id"})
    ls["gap_days"] = ls.lead_mean - ls.lead_master
    st.markdown("**Actual lead time vs master data**")
    st.dataframe(ls, hide_index=True, width="stretch", column_config={"lead_mean": NUM, "lead_sd": NUM, "lead_master": NUM, "gap_days": NUM})
    st.markdown("#### How good is the site's consumption plan?")
    howto("<b>WAPE</b> = Σ|actual − plan| ÷ Σ actual. <b>Bias</b> > 0: the plan over-states use (sites slower than planned). "
          "<b>MASE</b> < 1: the plan beats a naive 'same as last week' forecast.")
    fa = I.forecast_accuracy(M["led_clean"], d["plan_weekly"], M["items"], DATA_DATE)
    st.dataframe(fa, hide_index=True, width="stretch", column_config={"actual_total": NUM, "plan_total": NUM, "wape_%": NUM, "bias_%": NUM,
                                                                      "mase": st.column_config.NumberColumn(format="%.2f")})


def _checks(M, service, cap):
    howto("Planning is only as good as the master data and the ledger. These checks run on every movement. Flagged rows are kept in the "
          "ledger (nothing is deleted) but left out of the planning numbers until fixed.")
    ch = I.data_checks(M["items"], M["led"], M["d"]["locations"], DATA_DATE)
    st.dataframe(ch, hide_index=True, width="stretch")
    dirty = I.policy(M["plan_items"], M["d"]["plan_weekly"], M["d"]["deliveries"], M["led"], DATA_DATE, M["req"], service, cap)
    a = dirty.set_index("item_id").safety_stock.get("CEM-OPC")
    b = M["pol"].set_index("item_id").safety_stock.get("CEM-OPC")
    if a and b:
        st.info(f"Why it matters: with the bag-as-tonne entries left in, cement safety stock would be **{a:,.0f} t** instead of **{b:,.0f} t**: "
                f"about {money((a - b) * 165)} of cement bought and stored for nothing.")
    flagged = M["led"][I.suspect_rows(M["led"])]
    if len(flagged):
        st.markdown("**Rows left out of planning until fixed**")
        st.dataframe(flagged[["txn_id", "date", "item_id", "txn_type", "from_loc", "to_loc", "qty", "unit", "project_id", "note"]], hide_index=True, width="stretch",
                     column_config={"date": st.column_config.DateColumn(format="DD MMM YYYY")})


# ---------------------------------------------------------------- company, project, network
DATE = st.column_config.DateColumn(format="DD MMM YYYY")


def _company(M):
    howto("The company's whole material position. <b>On hand</b> by tier (hub, sub-stores, site stores) + <b>in transit</b> (shipped from the "
          "factory) + <b>on order</b> (not shipped yet) − <b>allocated</b> to the remaining work on all six buildings = <b>free</b> stock. "
          "Negative free stock means more must be bought. The forecast runs the material plan forward 26 weeks.")
    pos, fc, items, d = M["pos"], M["fc"], M["items"], M["d"]
    it = items.set_index("item_id")
    pos = pos.assign(value_in_transit=pos.in_transit * pos.item_id.map(it.unit_cost_usd), value_on_order=pos.on_order * pos.item_id.map(it.unit_cost_usd))
    k = st.columns(5)
    k[0].metric("On hand, all stores", money(pos.value_on_hand.sum()))
    k[1].metric("In transit", money(pos.value_in_transit.sum()))
    k[2].metric("On order, not shipped", money(pos.value_on_order.sum()))
    short = pos[pos.free < -1e-6]
    k[3].metric("Items short overall", len(short), help="Free stock below zero: stock + pipeline does not cover the remaining work")
    k[4].metric("Items forecast to run out", int(pos.forecast_runs_out.notna().sum()))
    hist = value_history(M)
    fut = fc.groupby("week").value_usd.sum().reset_index().rename(columns={"week": "date"})
    fut["series"] = "Forecast (all stores)"
    tot = hist.groupby("date").value_usd.sum().reset_index()
    tot["series"] = "Actual (all stores)"
    fut = pd.concat([tot.tail(1).assign(series="Forecast (all stores)"), fut], ignore_index=True)   # join the lines
    both = pd.concat([tot, fut], ignore_index=True)
    ch = alt.Chart(both).mark_line(strokeWidth=2.5).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value_usd:Q", title="Stock value ($)"),
        color=alt.Color("series:N", scale=alt.Scale(domain=["Actual (all stores)", "Forecast (all stores)"], range=[SERIES[0], SERIES[3]]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.series == "Forecast (all stores)", alt.value([6, 4]), alt.value([0])),
        tooltip=["series", "date:T", alt.Tooltip("value_usd:Q", format="$,.0f")])
    tiers = alt.Chart(hist).mark_area(opacity=0.25).encode(x="date:T", y=alt.Y("value_usd:Q", stack=True),
                                                         color=alt.Color("tier:N", scale=TIER_COLOR, legend=None))
    st.altair_chart((tiers + ch).properties(height=260, title="Company stock value: history by tier (shaded) and 26-week forecast (dashed)"),
                    width="stretch")
    fam = st.multiselect("Families", sorted(pos.family.unique()), sorted(pos.family.unique()), key="co_fam")
    st.dataframe(pos[pos.family.isin(fam)][["item_id", "item", "family", "unit", "hub", "sub_stores", "sites", "on_hand", "in_transit", "on_order",
                                           "allocated_to_projects", "free", "value_on_hand", "weeks_of_supply", "forecast_runs_out"]],
                 hide_index=True, width="stretch",
                 column_config={c: NUM for c in ["hub", "sub_stores", "sites", "on_hand", "in_transit", "on_order", "allocated_to_projects", "free", "weeks_of_supply"]}
                 | {"value_on_hand": DOLLAR, "forecast_runs_out": DATE})
    if (pos.sub_stores < 0).any():
        st.caption("A negative sub-store balance is a ledger error (a transfer posted before its receipt): see Data checks.")
    lab = dict(zip(items.item_id, items.item_id + " · " + items.name))
    item = st.selectbox("Forecast for one item", list(pos.item_id), format_func=lab.get, key="co_item")
    f = fc[fc.item_id == item]
    base = alt.Chart(f).encode(x=alt.X("week:T", title=None))
    st.altair_chart((base.mark_bar(color=SERIES[1], opacity=0.7).encode(y=alt.Y("gross:Q", title=f"{it.unit[item]} per week / on hand"))
                     + base.mark_line(color=SERIES[0], strokeWidth=2, point=True).encode(y="projected_on_hand:Q",
                                                                                          tooltip=["week:T", alt.Tooltip("projected_on_hand:Q", format=",.0f")])
                     + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=STATUS["critical"]).encode(y="y:Q"))
                    .properties(height=220, title="Need per week (bars) and projected stock, all stores (line)"), width="stretch")


def value_history(M):
    return I.value_history(M["led_clean"], M["d"]["locations"], M["items"], DATA_DATE)


def _project(M, service, cap):
    howto("One building at a time. Its own site stock, its share of the shared hub and sub-store stock (in proportion to its remaining need), what "
          "is still needed by part of the building, and a <b>site-level</b> safety stock and reorder point sized on this building's own "
          "forecast error and the 3-day sub-store-to-site lead time. The hub keeps the pooled buffer; the site keeps a small one.")
    import schedule as S_
    pids = list(_projects().project_id)
    names = dict(zip(_projects().project_id, _projects().short_name))
    pid = st.selectbox("Building", pids, format_func=lambda p: f"{p} · {names.get(p, p)}", key="pv_pid")
    pv = I.project_view(pid, M["plan_items"], M["bal"], M["req"], M["led_clean"], M["d"]["plan_weekly"], DATA_DATE, service, cap,
                        S_.template("datacenter"))
    if pv.empty:
        st.success("Nothing left to supply and nothing in the site store for this building.")
        return
    eq = M["eq"][M["eq"].project_id == pid]
    k = st.columns(6)
    k[0].metric("Stock at this site", money(pv.site_value.sum()))
    k[1].metric("Material still to supply", money(pv.remaining_need_value.sum()))
    k[2].metric("Excess at site", money(pv[pv.remaining_need.gt(0) | pv.item_id.isin(M["items"][M["items"].family != "Equipment"].item_id)].excess_value.sum()),
                help="Bulk material at the site beyond what this building still needs: transfer it to another building or return it to the sub-store")
    k[3].metric("Carrying cost per day (site)", money(pv.carry_per_day.sum()))
    k[4].metric("Call-offs needed now", int(pv.call_off_now.sum()), help="Work needs the item within 2 weeks and site stock would fall below its reorder point")
    k[5].metric("Equipment late vs need date", int(eq.status.str.startswith(("LATE", "NOT")).sum()) if len(eq) else 0)
    by = pv.groupby("part_of_building").agg(remaining=("remaining_need_value", "sum"), at_site=("site_value", "sum")).reset_index()
    m = by.melt(id_vars="part_of_building", var_name="what", value_name="usd")
    m["what"] = m.what.map({"remaining": "Still to supply", "at_site": "At site now"})
    st.altair_chart(alt.Chart(m).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("part_of_building:N", title=None, sort="-x", axis=alt.Axis(labelLimit=220)), x=alt.X("usd:Q", title="$"), yOffset="what:N",
        color=alt.Color("what:N", scale=alt.Scale(range=[SERIES[0], SERIES[2]]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["part_of_building", "what", alt.Tooltip("usd:Q", format="$,.0f")]).properties(height=alt.Step(16), title="By part of the building"),
        width="stretch")
    st.dataframe(pv.sort_values(["part_of_building", "task_id"]), hide_index=True, width="stretch",
                 column_config={c: NUM for c in ["remaining_need", "at_site", "share_of_shared_stock", "weekly_need", "need_next_2wk", "site_safety_stock",
                                                 "site_reorder_point", "weeks_of_supply", "excess_at_site"]}
                 | {"covered_by_site_%": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                    "site_value": DOLLAR, "remaining_need_value": DOLLAR, "excess_value": DOLLAR, "carry_per_day": st.column_config.NumberColumn(format="$%,.0f")})
    bulk = pv[pv.site_reorder_point.notna() & (pv.remaining_need > 0)]
    if len(bulk):
        lab = dict(zip(bulk.item_id, bulk.item_id + " · " + bulk["item"]))
        c1, c2 = st.columns([2, 1])
        item = c1.selectbox("Site plan for item", list(lab), format_func=lab.get, key="pv_item")
        r = bulk.set_index("item_id").loc[item]
        lot = c2.number_input(f"Call-off size ({r.unit})", 0.0, 1e7, float(max(1.0, round(r.weekly_need * 2, -1))), key="pv_lot")
        sp = I.site_plan(pid, item, M["req"], M["bal"], r.site_reorder_point, r.site_safety_stock, lot, DATA_DATE)
        base = alt.Chart(sp).encode(x=alt.X("week:T", title=None))
        st.altair_chart((base.mark_bar(color=SERIES[1], opacity=0.7).encode(y=alt.Y("need:Q", title=r.unit))
                         + base.mark_bar(color=SERIES[2], opacity=0.9, width=6).encode(y="call_off:Q", tooltip=["week:T", alt.Tooltip("call_off:Q", format=",.0f")])
                         + base.mark_line(color=SERIES[0], strokeWidth=2, point=True).encode(y="site_stock:Q", tooltip=["week:T", alt.Tooltip("site_stock:Q", format=",.0f")])
                         + alt.Chart(pd.DataFrame({"y": [r.site_reorder_point]})).mark_rule(color=STATUS["warning"], strokeDash=[5, 4]).encode(y="y:Q"))
                        .properties(height=230, title="Need (orange), call-offs from the sub-store (green), site stock (line), reorder point (dashed)"),
                        width="stretch")
    if len(eq):
        st.markdown("**Long-lead equipment for this building**")
        pp = M["pipe"][M["pipe"].project_id == pid][["po_id", "item_id", "current_stage", "eta_p50", "eta_p90", "need_date", "chance_late_%"]]
        st.dataframe(eq.merge(pp, on="item_id", how="left", suffixes=("", "_pipe")), hide_index=True, width="stretch",
                     column_config={"delivery_due": DATE, "need_date": DATE, "need_date_pipe": None, "eta_p50": DATE, "eta_p90": DATE, "value_usd": DOLLAR,
                                    "chance_late_%": st.column_config.NumberColumn(format="%.0f%%")})
    st.markdown("**All six buildings side by side**")
    rows = []
    for p in _projects().project_id:
        v = I.project_view(p, M["plan_items"], M["bal"], M["req"], M["led_clean"], M["d"]["plan_weekly"], DATA_DATE, service, cap, S_.template("datacenter"))
        e = M["eq"][M["eq"].project_id == p]
        rows.append({"building": f"{p} · {names.get(p, p)}", "site_stock": v.site_value.sum() if len(v) else 0.0,
                     "excess_at_site": v[~v.item_id.isin(M["items"][M["items"].family == "Equipment"].item_id)].excess_value.sum() if len(v) else 0.0,
                     "still_to_supply": v.remaining_need_value.sum() if len(v) else 0.0, "carry_per_day": v.carry_per_day.sum() if len(v) else 0.0,
                     "call_offs_now": int(v.call_off_now.sum()) if len(v) else 0,
                     "equipment_late": int(e.status.str.startswith(("LATE", "NOT")).sum()) if len(e) else 0})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"site_stock": DOLLAR, "excess_at_site": DOLLAR, "still_to_supply": DOLLAR, "carry_per_day": st.column_config.NumberColumn(format="$%,.0f")})


def _network(M):
    howto("Where every product is made, and how long each step takes on its way to West Texas: manufacturing, inland to the port, port dwell, "
          "ocean, US port dwell, customs, truck or rail. Lead time = sum of the stages; uncertainty adds as variance, so you can see <b>which stage "
          "makes the lead time unpredictable</b>. Every open order shows where it is now and its likely arrival (P50 / P90, Monte Carlo) against "
          "the date the schedule needs it. Second-tier inputs show what the factory itself depends on.")
    d, items = M["d"], M["items"]
    fac = d["factories"]
    hub = d["locations"].set_index("location_id").loc["HUB-ODS"]
    arcs = fac.assign(dst_lat=hub.lat, dst_lon=hub.lon, color=fac.country.map(lambda c: [57, 135, 229] if c == "USA" else [217, 89, 38]))
    layers = [pdk.Layer("ArcLayer", arcs, get_source_position=["lon", "lat"], get_target_position=["dst_lon", "dst_lat"], get_source_color="color",
                        get_target_color=[25, 158, 112], get_width=2, pickable=True),
              pdk.Layer("ScatterplotLayer", fac, get_position=["lon", "lat"], get_radius=60000, get_fill_color=[217, 89, 38], pickable=True)]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=28, longitude=-20, zoom=0.9),
                             map_style=pdk.map_styles.CARTO_LIGHT, tooltip={"text": "{factory}, {city} ({country})\nMakes: {items}\n{capacity_note}"}),
                    height=380)
    st.dataframe(fac.merge(d["suppliers"][["supplier_id", "name"]], on="supplier_id").rename(columns={"name": "supplier"})[
        ["factory_id", "supplier", "factory", "city", "country", "items", "capacity_note"]], hide_index=True, width="stretch")
    st.markdown("##### Lead time, stage by stage")
    pi = M["plan_items"]
    lab = dict(zip(pi.item_id, pi.item_id + " · " + pi.name + " (" + pi.origin + ")"))
    item = st.selectbox("Product", list(lab), index=list(lab).index("XFMR") if "XFMR" in lab else 0, format_func=lab.get, key="nw_item")
    x, tot = I.lead_breakdown(item, d["lanes"])
    ls = I.lead_stats(items, d["deliveries"])
    k = st.columns(4)
    k[0].metric("Planned lead time (mean)", f"{tot['mean']:.0f} days")
    k[1].metric("Spread (1 sd)", f"±{tot['sd']:.0f} days")
    k[2].metric("P90 lead time", f"{tot['p90']:.0f} days", help="Plan the order date on this, not the mean, for critical items")
    k[3].metric("Actual from past deliveries", f"{ls.lead_mean.get(item, float('nan')):.0f} days", f"{int(ls.lead_obs.get(item, 0))} deliveries", delta_color="off")
    x["start"] = x.mean_days.cumsum() - x.mean_days
    x["end"] = x.mean_days.cumsum()
    g = alt.Chart(x).mark_bar(cornerRadius=3, height=22).encode(
        x=alt.X("start:Q", title="Days from order"), x2="end:Q", y=alt.Y("stage:N", sort=list(x.stage), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("share_of_uncertainty_%:Q", scale=alt.Scale(range=["#c4c7c5", SERIES[1]]), legend=alt.Legend(title="share of uncertainty %")),
        tooltip=["stage", alt.Tooltip("mean_days:Q", format=".0f"), alt.Tooltip("sd_days:Q", format=".1f"), alt.Tooltip("share_of_uncertainty_%:Q", format=".0f")])
    st.altair_chart(g.properties(height=alt.Step(28)), width="stretch")
    top = x.sort_values("var", ascending=False).iloc[0]
    st.caption(f"'{top.stage}' carries {top['share_of_uncertainty_%']:.0f}% of the uncertainty. That is where a buffer, a progress inspection or "
               "a second source does the most good.")
    t2 = d["tier2"][d["tier2"].item_id == item]
    if len(t2):
        st.markdown("**What the factory depends on (tier 2)**")
        st.dataframe(t2, hide_index=True, width="stretch")
        if t2.single_source.any():
            st.warning("Single-source tier-2 input: a problem at that one sub-supplier stops this product regardless of the factory's own capacity. "
                       "Ask the supplier for its sub-supplier plan.")
    st.markdown("##### Every open order: where it is and when it will arrive")
    pp = M["pipe"].merge(items[["item_id", "name", "origin"]], on="item_id")
    pp["project_id"] = pp.project_id.where(pp.project_id.astype(str).str.startswith("WTX"), "network (hub)")
    flt = st.multiselect("Show", ["Likely late vs need", "At risk", "Past promised date", "OK"], ["Likely late vs need", "At risk", "Past promised date", "OK"], key="nw_flt")
    pp = pp[pp.flag.isin(flt)]
    st.dataframe(pp[["po_id", "item_id", "name", "origin", "project_id", "qty", "current_stage", "stage_no", "days_in_stage", "promised_due", "eta_p50",
                     "eta_p90", "need_date", "margin_p50_days", "chance_late_%", "flag"]].sort_values("chance_late_%", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"promised_due": DATE, "eta_p50": DATE, "eta_p90": DATE, "need_date": DATE, "qty": NUM, "days_in_stage": NUM,
                                "chance_late_%": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    st.caption("Need date: equipment = the building's task start from the CPM forecast; bulk replenishment = the week network stock would fall "
               "below safety stock without that order. ETA = today + the remaining time in the current stage + every later stage, simulated "
               "4,000 times with each stage's own spread.")
