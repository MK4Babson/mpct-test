"""
Build the synthetic supply-chain dataset for the Materials & supply chain page (data/scm/).

Everything here is SYNTHETIC and labelled as such in the app. It is shaped like a real material system:
  items.csv          item master: base unit, alternate unit and conversion, cost, lead time, origin, currency, task
  locations.csv      regional hub -> campus sub-stores -> building site stores, with coordinates
  suppliers.csv      fictional suppliers by country and currency
  ledger.csv         every stock movement (receipt, transfer, issue, adjustment) from Jan 2026 to the data date
  deliveries.csv     supplier deliveries: promised vs actual date and quantity (for OTIF and lead-time variability)
  open_pos.csv       open purchase orders with payment milestones in the supplier's currency (FX exposure)
  plan_weekly.csv    the weekly consumption plan the site committed to (for forecast accuracy: WAPE, bias, MASE)

Consumption follows each building's CPM baseline (same task templates and contract window as the Schedule page),
so the material plan and the schedule are driven by the same dates.

A few errors are planted on purpose so the data-quality checks have something real to find:
  cement issues keyed in bags but labelled tonnes, one duplicated transaction, a transfer posted before its receipt,
  a duplicate item in the master, and a supplier lead time that has not been reviewed for over a year.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import schedule as S  # noqa: E402

OUT = ROOT / "data" / "scm"
DATA_DATE = pd.Timestamp("2026-08-31")
FX_PER_USD = {"USD": 1.0, "EUR": 0.90, "KRW": 1380.0, "INR": 85.0, "CNY": 7.2, "JPY": 150.0, "MXN": 18.5}

# item, name, family, base unit, alt unit, alt per base, usd per base unit, qty per building (scale 1), task, origin,
# currency, supplier, lead mean, lead sd (days), shelf life days, flow (hub = via hub/sub-store, direct = to site),
# lot size (base units), storage cost % of value per year, critical (long lead / critical path)
ITEMS = [
    ("CEM-OPC", "Portland cement Type I/II", "Concrete", "t", "bag (50 kg)", 20.0, 165.0, 4500, "A05", "USA", "USD", "SUP-TXC", 5, 2, 90, "hub", 600, 0.04, False),
    ("AGG-57", "No. 57 crushed aggregate", "Concrete", "t", "", 1.0, 28.0, 9000, "A05", "USA", "USD", "SUP-PBA", 3, 1, 9999, "direct", 1500, 0.02, False),
    ("RBR-5", "Rebar #5 Grade 60", "Steel", "t", "", 1.0, 1150.0, 1400, "A05", "USA", "USD", "SUP-SWM", 21, 7, 9999, "hub", 250, 0.03, False),
    ("STL-W", "Structural steel W-shapes", "Steel", "t", "", 1.0, 2400.0, 3200, "A07", "USA", "USD", "SUP-SWM", 60, 15, 9999, "direct", 800, 0.03, False),
    ("CT-24", "Cable tray 24 in ladder", "Electrical", "ft", "", 1.0, 30.0, 18000, "A10", "USA", "USD", "SUP-GCE", 30, 8, 9999, "hub", 6000, 0.03, False),
    ("MV-500", "MV cable 15 kV 500 kcmil", "Electrical", "ft", "reel (2,500 ft)", 0.0004, 22.0, 42000, "A11", "Korea", "KRW", "SUP-HNC", 70, 20, 9999, "hub", 15000, 0.03, False),
    ("CND-4", "Conduit 4 in PVC Sch 40", "Electrical", "ft", "", 1.0, 4.0, 30000, "A09", "USA", "USD", "SUP-GCE", 10, 3, 9999, "hub", 10000, 0.03, False),
    ("LV-CU", "LV copper building wire", "Electrical", "ft", "", 1.0, 2.5, 120000, "A11", "USA", "USD", "SUP-GCE", 21, 6, 9999, "hub", 40000, 0.03, False),
    ("BUS-4K", "Busway 4000 A", "Electrical", "ft", "", 1.0, 450.0, 1500, "A12", "Mexico", "MXN", "SUP-MXE", 56, 14, 9999, "hub", 500, 0.03, False),
    ("PIPE-CHW", "Steel pipe, chilled water (ERW)", "Mechanical", "ft", "", 1.0, 95.0, 9000, "A20", "India", "INR", "SUP-INS", 75, 25, 9999, "hub", 3000, 0.03, False),
    ("DUCT", "Galvanized ductwork", "Mechanical", "lb", "", 1.0, 3.2, 60000, "A22", "USA", "USD", "SUP-GCE", 14, 4, 9999, "hub", 20000, 0.03, False),
    ("SWG", "MV switchgear lineup", "Equipment", "ea", "", 1.0, 420000.0, 8, "A14", "Mexico", "USD", "SUP-MXE", 300, 45, 9999, "direct", 8, 0.02, True),
    ("XFMR", "Transformer 2.5 MVA pad-mount", "Equipment", "ea", "", 1.0, 380000.0, 8, "A14", "Korea", "KRW", "SUP-HNC", 365, 60, 9999, "direct", 8, 0.02, True),
    ("UPS", "UPS module 1.2 MW", "Equipment", "ea", "", 1.0, 260000.0, 16, "A15", "India", "INR", "SUP-INE", 200, 35, 9999, "direct", 16, 0.02, True),
    ("GEN", "Diesel generator 3 MW", "Equipment", "ea", "", 1.0, 1400000.0, 8, "A17", "USA", "USD", "SUP-USG", 340, 50, 9999, "direct", 8, 0.02, True),
    ("CHL", "Air-cooled chiller 500 ton", "Equipment", "ea", "", 1.0, 450000.0, 10, "A19", "Germany", "EUR", "SUP-DEK", 280, 40, 9999, "direct", 10, 0.02, True),
    ("CRAH", "CRAH unit 150 kW", "Equipment", "ea", "", 1.0, 45000.0, 40, "A23", "China", "CNY", "SUP-CNA", 120, 30, 9999, "direct", 40, 0.02, False),
    ("CDU", "Coolant distribution unit", "Equipment", "ea", "", 1.0, 90000.0, 24, "A24", "Japan", "JPY", "SUP-JPC", 150, 25, 9999, "direct", 24, 0.02, False),
]
ICOLS = ["item_id", "name", "family", "unit", "alt_unit", "alt_per_unit", "unit_cost_usd", "qty_per_building", "task_id", "origin",
         "currency", "supplier_id", "lead_days", "lead_sd", "shelf_life_days", "flow", "lot_size", "storage_rate", "critical"]

SUPPLIERS = [
    ("SUP-TXC", "Lone Star Cement (fictional)", "USA", "USD", "DAP site", 30),
    ("SUP-PBA", "Permian Basin Aggregates (fictional)", "USA", "USD", "DAP site", 30),
    ("SUP-SWM", "Southwest Mill & Fab (fictional)", "USA", "USD", "FCA mill", 45),
    ("SUP-GCE", "Gulf Coast Electrical Supply (fictional)", "USA", "USD", "DAP hub", 45),
    ("SUP-HNC", "Hanbit Cable & Power (fictional)", "Korea", "KRW", "FOB Busan", 60),
    ("SUP-MXE", "Monterrey Electro (fictional)", "Mexico", "MXN", "FCA Monterrey", 45),
    ("SUP-INS", "Deccan Steel Tubes (fictional)", "India", "INR", "FOB Mundra", 60),
    ("SUP-INE", "Narmada Power Systems (fictional)", "India", "INR", "FOB Nhava Sheva", 60),
    ("SUP-USG", "Prairie Power Generation (fictional)", "USA", "USD", "DAP site", 30),
    ("SUP-DEK", "Rheinland Kaeltetechnik (fictional)", "Germany", "EUR", "FOB Hamburg", 60),
    ("SUP-CNA", "Pearl River Air Systems (fictional)", "China", "CNY", "FOB Shenzhen", 60),
    ("SUP-JPC", "Kansai Coolant Systems (fictional)", "Japan", "JPY", "FOB Kobe", 60),
]

# ---------------------------------------------------------------- supply network (synthetic, plausible geography)
# factory_id, supplier_id, factory name, city, country, lat, lon, items made, monthly capacity note
FACTORIES = [
    ("F-TXC-MID", "SUP-TXC", "Kiln 2", "Midlothian, TX", "USA", 32.48, -96.99, "CEM-OPC", "cement kiln, ~90,000 t/month"),
    ("F-PBA-BSP", "SUP-PBA", "Quarry 7", "Big Spring, TX", "USA", 32.25, -101.48, "AGG-57", "quarry, ~150,000 t/month"),
    ("F-SWM-FTW", "SUP-SWM", "Mill & fab shop", "Fort Worth, TX", "USA", 32.75, -97.33, "RBR-5;STL-W", "rebar mill + structural fab"),
    ("F-GCE-HOU", "SUP-GCE", "Distribution center", "Houston, TX", "USA", 29.76, -95.37, "CT-24;CND-4;LV-CU;DUCT", "stocking distributor (makes nothing itself)"),
    ("F-HNC-GUM", "SUP-HNC", "Cable plant", "Gumi", "Korea", 36.12, 128.34, "MV-500", "MV cable lines"),
    ("F-HNC-CHW", "SUP-HNC", "Transformer works", "Changwon", "Korea", 35.23, 128.68, "XFMR", "transformer winding + test bay"),
    ("F-MXE-MTY", "SUP-MXE", "Switchgear plant", "Monterrey", "Mexico", 25.69, -100.32, "SWG;BUS-4K", "switchgear + busway assembly"),
    ("F-INS-KUT", "SUP-INS", "ERW tube mill", "Kutch, Gujarat", "India", 23.07, 70.13, "PIPE-CHW", "ERW pipe mill near Mundra port"),
    ("F-INE-VAD", "SUP-INE", "UPS assembly", "Vadodara, Gujarat", "India", 22.31, 73.18, "UPS", "UPS assembly + battery cabinet integration"),
    ("F-USG-IND", "SUP-USG", "Genset assembly", "Lafayette, IN", "USA", 40.42, -86.88, "GEN", "genset packaging + load test"),
    ("F-DEK-KOL", "SUP-DEK", "Chiller plant", "Cologne", "Germany", 50.94, 6.96, "CHL", "chiller assembly + run test"),
    ("F-CNA-FOS", "SUP-CNA", "Air-handling plant", "Foshan, Guangdong", "China", 23.02, 113.12, "CRAH", "CRAH assembly"),
    ("F-JPC-OSA", "SUP-JPC", "CDU plant", "Osaka", "Japan", 34.69, 135.50, "CDU", "CDU assembly + pressure test"),
]
# lane by origin: ordered stages (name, mean days, sd days). Manufacturing = item lead time minus the rest.
LANES = {
    "USA": [("Manufacturing / pick", None, None), ("Truck to West Texas", 3, 1)],
    "Mexico": [("Manufacturing", None, None), ("Truck to Laredo border", 2, 0.5), ("Border crossing + customs", 3, 2), ("Truck to West Texas", 3, 1)],
    "Korea": [("Manufacturing", None, None), ("Inland to Busan", 2, 1), ("Origin port dwell", 4, 2), ("Ocean: Busan to Houston (Panama Canal)", 32, 5),
              ("Houston port dwell", 4, 3), ("US customs clearance", 3, 2), ("Truck/rail to West Texas", 4, 1.5)],
    "India": [("Manufacturing", None, None), ("Inland to Mundra", 2, 1), ("Origin port dwell", 5, 3), ("Ocean: Mundra to Houston (Suez or Cape)", 38, 9),
              ("Houston port dwell", 4, 3), ("US customs clearance", 4, 3), ("Truck/rail to West Texas", 4, 1.5)],
    "Germany": [("Manufacturing", None, None), ("Inland to Hamburg", 2, 1), ("Origin port dwell", 3, 1.5), ("Ocean: Hamburg to Houston", 20, 3),
                ("Houston port dwell", 4, 3), ("US customs clearance", 3, 2), ("Truck/rail to West Texas", 4, 1.5)],
    "China": [("Manufacturing", None, None), ("Inland to Shekou", 1, 0.5), ("Origin port dwell", 4, 2), ("Ocean: Shekou to Houston (Panama Canal)", 33, 5),
              ("Houston port dwell", 4, 3), ("US customs clearance", 5, 4), ("Truck/rail to West Texas", 4, 1.5)],
    "Japan": [("Manufacturing", None, None), ("Inland to Kobe", 1, 0.5), ("Origin port dwell", 3, 1.5), ("Ocean: Kobe to Houston (Panama Canal)", 30, 4),
              ("Houston port dwell", 4, 3), ("US customs clearance", 3, 2), ("Truck/rail to West Texas", 4, 1.5)],
}
# key second-tier inputs: item, input, tier-2 maker (fictional), country, single source?
TIER2 = [
    ("XFMR", "Grain-oriented electrical steel (core)", "Pohang Electrical Steel (fictional)", "Korea", True),
    ("XFMR", "Copper winding wire", "Ulsan Copper (fictional)", "Korea", False),
    ("SWG", "Vacuum interrupters", "Rhine Interrupter GmbH (fictional)", "Germany", True),
    ("SWG", "Protection relays", "Guadalajara Controls (fictional)", "Mexico", False),
    ("UPS", "Lithium-ion battery cells", "Ningde Cells (fictional)", "China", True),
    ("UPS", "IGBT power modules", "Kyushu Semiconductor (fictional)", "Japan", False),
    ("GEN", "Diesel engine block", "Columbus Engine Works (fictional)", "USA", True),
    ("GEN", "Alternator", "Monterrey Alternators (fictional)", "Mexico", False),
    ("CHL", "Screw compressors", "Brno Compressors (fictional)", "Czechia", True),
    ("CRAH", "EC fans", "Suzhou Fan Motors (fictional)", "China", False),
    ("CDU", "Plate heat exchangers", "Lund Thermal (fictional)", "Sweden", False),
    ("MV-500", "Copper rod", "Onsan Smelter (fictional)", "Korea", False),
    ("PIPE-CHW", "Hot-rolled coil", "Hazira Steel (fictional)", "India", False),
    ("BUS-4K", "Aluminum bus bar", "Veracruz Aluminum (fictional)", "Mexico", False),
]

CAMPUS = {"Mesa": ("SUB-MESA", 32.383, -100.858), "Ridge": ("SUB-RIDGE", 33.124, -101.575), "Plains": ("SUB-PLAINS", 34.423, -101.797)}
HUB = ("HUB-ODS", "Odessa regional hub (fictional)", 31.846, -102.368)


def _rng(s):
    return np.random.default_rng(sum(map(ord, s)) + 7)


def baseline_windows(project_controls):
    """Baseline start/finish (dates) of every template task for every project, fitted to the contract window."""
    t0 = S.template("datacenter")
    base = S.cpm(t0.assign(dur=t0.m))
    rows = []
    for r in project_controls.itertuples():
        st, fin = pd.Timestamp(r.planned_start), pd.Timestamp(r.planned_finish)
        k = max(1.0, (fin - st).days) / max(1.0, base.attrs["finish"])
        t = t0.copy()
        t["dur"] = t.m * k
        c = S.cpm(t)
        for x in c.itertuples():
            rows.append({"project_id": r.project_id, "task_id": x.id, "bl_start": st + pd.Timedelta(days=float(x.es)),
                         "bl_finish": st + pd.Timedelta(days=float(x.ef))})
    return pd.DataFrame(rows)


def main(src=ROOT / "data" / "raw"):
    OUT.mkdir(parents=True, exist_ok=True)
    items = pd.DataFrame(ITEMS, columns=ICOLS)
    items["lead_time_reviewed"] = "2026-06-30"
    items.loc[items.item_id == "PIPE-CHW", "lead_time_reviewed"] = "2025-03-31"   # planted: stale lead time
    dup = items[items.item_id == "CEM-OPC"].copy()
    dup["item_id"], dup["name"] = "CEM-OPC-B", "Cement OPC Type I-II"              # planted: duplicate item
    items_out = pd.concat([items, dup], ignore_index=True)
    items_out.to_csv(OUT / "items.csv", index=False)
    pd.DataFrame(SUPPLIERS, columns=["supplier_id", "name", "country", "currency", "incoterm", "payment_days"]).to_csv(OUT / "suppliers.csv", index=False)

    projects = pd.read_csv(src / "projects.csv")
    controls = pd.read_csv(src / "project_controls.csv")
    camp_of = {p: c.split()[0] for p, c in zip(projects.project_id, projects.campus)}
    scale = dict(zip(projects.project_id, projects.scale))

    locs = [{"location_id": HUB[0], "name": HUB[1], "tier": "hub", "campus": "", "project_id": "", "lat": HUB[2], "lon": HUB[3], "capacity_m2": 12000}]
    for c, (sid, la, lo) in CAMPUS.items():
        locs.append({"location_id": sid, "name": f"{c} campus sub-store", "tier": "sub-store", "campus": c, "project_id": "", "lat": la, "lon": lo, "capacity_m2": 4000})
    for p in projects.itertuples():
        locs.append({"location_id": f"SITE-{p.project_id}", "name": f"{p.short_name} site store", "tier": "site", "campus": camp_of[p.project_id],
                     "project_id": p.project_id, "lat": p.latitude, "lon": p.longitude, "capacity_m2": 1500})
    pd.DataFrame(locs).to_csv(OUT / "locations.csv", index=False)

    win = baseline_windows(controls)
    win.to_csv(OUT / "baseline_windows.csv", index=False)
    perf = {"WTX-A1": 1.00, "WTX-A2": 0.97, "WTX-B1": 0.93, "WTX-B2": 0.95, "WTX-C1": 0.82, "WTX-C2": 0.90}   # C1 runs slow

    weeks = pd.date_range("2026-01-05", DATA_DATE + pd.Timedelta(days=160), freq="W-MON")
    plan_rows, ledger, deliv, open_pos = [], [], [], []
    tid = [0]

    def txn(date, item, frm, to, qty, kind, project="", unit=None, ref=""):
        tid[0] += 1
        ledger.append({"txn_id": f"T{tid[0]:06d}", "date": pd.Timestamp(date).normalize(), "item_id": item, "from_loc": frm, "to_loc": to,
                       "qty": round(float(qty), 3), "unit": unit or units[item], "txn_type": kind, "project_id": project, "ref": ref})

    units = dict(zip(items.item_id, items.unit))
    for it in items.itertuples():
        rng = _rng(it.item_id)
        for p in projects.project_id:
            w = win[(win.project_id == p) & (win.task_id == it.task_id)].iloc[0]
            q = it.qty_per_building * scale[p]
            if it.unit == "ea":
                q = float(max(1, round(q)))
            bs, bf = w.bl_start, w.bl_finish
            dur_w = max(1, (bf - bs).days / 7)
            # weekly plan (what the site committed to)
            for wk in weeks:
                overlap = max(0.0, (min(wk + pd.Timedelta(days=7), bf) - max(wk, bs)).days) / 7
                if overlap > 0:
                    plan_rows.append({"week": wk, "project_id": p, "item_id": it.item_id, "planned_qty": q * overlap / dur_w})
            # actual issues up to the data date: plan x performance x noise, capped at total
            site = f"SITE-{p}"
            sub = CAMPUS[camp_of[p]][0]
            issued = 0.0
            act_weeks = [wk for wk in weeks if wk <= DATA_DATE]
            for wk in act_weeks:
                overlap = max(0.0, (min(wk + pd.Timedelta(days=7), bf) - max(wk, bs)).days) / 7
                if overlap <= 0 or issued >= q:
                    continue
                if it.flow == "direct" and it.family == "Equipment":
                    continue
                qty = min(q - issued, q * overlap / dur_w * perf[p] * float(rng.normal(1, 0.12)))
                if qty <= 0:
                    continue
                d = wk + pd.Timedelta(days=int(rng.integers(0, 5)))
                if d > DATA_DATE:
                    continue
                # replenish then issue
                if it.flow == "hub":
                    txn(d - pd.Timedelta(days=2), it.item_id, sub, site, qty * 1.02, "TRANSFER", p)
                else:
                    txn(d - pd.Timedelta(days=1), it.item_id, it.supplier_id, site, qty * 1.03, "RECEIPT", p, ref=f"PO-{it.item_id}-{p}")
                txn(d, it.item_id, site, "", qty, "ISSUE", p)
                issued += qty
            # equipment: delivered to site near the set task start, set (issued) during the task
            if it.family == "Equipment":
                promised = bs - pd.Timedelta(days=10)
                late = float(rng.gamma(2.0, it.lead_sd / 6)) - it.lead_sd / 4
                actual = promised + pd.Timedelta(days=int(round(late)))
                qty_d = q
                in_full = not (rng.random() < 0.12)
                if not in_full:
                    qty_d = max(1, round(q * 0.75))
                if actual <= DATA_DATE:
                    txn(actual, it.item_id, it.supplier_id, site, qty_d, "RECEIPT", p, ref=f"PO-{it.item_id}-{p}")
                    deliv.append({"po_id": f"PO-{it.item_id}-{p}", "item_id": it.item_id, "supplier_id": it.supplier_id, "project_id": p,
                                  "order_date": promised - pd.Timedelta(days=int(it.lead_days)), "promised_date": promised, "delivered_date": actual,
                                  "qty_ordered": q, "qty_delivered": qty_d})
                    setq = 0.0
                    for wk in act_weeks:
                        overlap = max(0.0, (min(wk + pd.Timedelta(days=7), bf) - max(wk, max(bs, actual))).days) / 7
                        if overlap > 0 and setq < qty_d:
                            k = min(qty_d - setq, round(q * overlap / dur_w * perf[p] + 0.49))
                            if k > 0:
                                txn(wk + pd.Timedelta(days=2), it.item_id, site, "", k, "ISSUE", p)
                                setq += k
                    if qty_d < q:
                        open_pos.append((it, p, q - qty_d, actual + pd.Timedelta(days=28), "balance of short shipment"))
                else:
                    open_pos.append((it, p, q, actual, "not yet delivered"))

    led = pd.DataFrame(ledger)
    # hub -> sub-store replenishment and supplier receipts at the hub (lots), built from what the sub-stores sent to sites
    hub_items = items[items.flow == "hub"].item_id
    for item in hub_items:
        it = items[items.item_id == item].iloc[0]
        rng = _rng(item + "hub")
        # opening stock at the hub
        txn("2026-01-02", item, "OPENING", HUB[0], it.lot_size * 0.5, "RECEIPT", ref="OPENING")
        for c, (sub, _, _) in CAMPUS.items():
            x = led[(led.item_id == item) & (led.from_loc == sub)]
            out = x.groupby(x.date.dt.to_period("M").dt.start_time).qty.sum()
            for d, qn in out.items():
                if qn > 0:   # monthly replenishment of the sub-store, a few days before the month starts
                    txn(d - pd.Timedelta(days=4), item, HUB[0], sub, qn * 1.03, "TRANSFER")
        led2 = pd.DataFrame(ledger)
        need = led2[(led2.item_id == item) & (led2.from_loc == HUB[0])].set_index("date").qty.sort_index()
        cum_need = need.cumsum()
        stock_in, k = it.lot_size * 0.5, 0
        for d, cn in cum_need.items():
            while stock_in < cn + it.lot_size * 0.15:
                k += 1
                promised = d - pd.Timedelta(days=7)
                late = int(round(float(rng.normal(0.15, 0.6)) * it.lead_sd))
                actual = promised + pd.Timedelta(days=late)
                q_ord = it.lot_size
                q_del = q_ord if rng.random() > 0.1 else round(q_ord * 0.85, 1)
                txn(min(actual, d - pd.Timedelta(days=1)), item, it.supplier_id, HUB[0], q_del, "RECEIPT", ref=f"PO-{item}-H{k:02d}")
                deliv.append({"po_id": f"PO-{item}-H{k:02d}", "item_id": item, "supplier_id": it.supplier_id, "project_id": "",
                              "order_date": promised - pd.Timedelta(days=int(it.lead_days)), "promised_date": promised,
                              "delivered_date": actual, "qty_ordered": q_ord, "qty_delivered": q_del})
                stock_in += q_del
        # one or two lots still open
        rem_plan = sum(r["planned_qty"] for r in plan_rows if r["item_id"] == item and r["week"] > DATA_DATE)
        if rem_plan > 0:
            n_open = int(np.ceil(rem_plan / it.lot_size))
            for j in range(min(n_open, 3)):
                open_pos.append((it, "", it.lot_size, DATA_DATE + pd.Timedelta(days=int(10 + 21 * j)), "hub replenishment"))

    # blanket orders with monthly call-offs for bulk items delivered straight to site (local suppliers)
    pw = pd.DataFrame(plan_rows)
    for it in items[(items.flow == "direct") & (items.family != "Equipment")].itertuples():
        fut = pw[(pw.item_id == it.item_id) & (pw.week > DATA_DATE)].copy()
        fut["month"] = fut.week.dt.to_period("M").dt.start_time
        for (p, mth), q in fut.groupby(["project_id", "month"]).planned_qty.sum().items():
            if q > 0:
                open_pos.append((it, p, q, max(mth, DATA_DATE + pd.Timedelta(days=3)), "blanket order call-off"))

    led = pd.DataFrame(ledger).sort_values(["date", "txn_id"]).reset_index(drop=True)
    # planted errors
    cem = led[(led.item_id == "CEM-OPC") & (led.txn_type == "ISSUE")].index[[5, 17, 30]]
    led.loc[cem, "qty"] = (led.loc[cem, "qty"] * 20).round(1)          # keyed in bags, labelled tonnes
    led.loc[cem, "note"] = "planted: bags keyed as tonnes"
    d = led.iloc[[120]].copy()
    d["note"] = "planted: duplicate"
    led = pd.concat([led, d], ignore_index=True)                       # duplicate transaction id
    early = led[(led.txn_type == "TRANSFER") & (led.from_loc == "SUB-RIDGE") & (led.item_id == "CT-24")].index[:1]
    led.loc[early, "qty"] = led.loc[early, "qty"] * 6                  # transfer bigger than stock -> negative balance
    led.loc[early, "note"] = "planted: transfer before receipt"
    # a few cement receipts at sites keyed in bags with the right unit (must convert, not an error)
    bag = led[(led.item_id == "CEM-OPC") & (led.txn_type == "TRANSFER")].index[2:6]
    led.loc[bag, "qty"] = (led.loc[bag, "qty"] * 20).round(0)
    led.loc[bag, "unit"] = "bag (50 kg)"
    led["note"] = led.get("note", pd.Series(index=led.index, dtype=object)).fillna("")
    led.sort_values(["date", "txn_id"]).to_csv(OUT / "ledger.csv", index=False)
    pd.DataFrame(deliv).to_csv(OUT / "deliveries.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(OUT / "plan_weekly.csv", index=False)

    # open purchase orders with payment milestones in supplier currency
    rows = []
    for i, (it, p, q, due, why) in enumerate(open_pos):
        amt_usd = q * it.unit_cost_usd
        fx = FX_PER_USD[it.currency]
        po = f"OPO-{i + 1:03d}"
        if it.family == "Equipment":
            ms = [("shipment", 0.6, due - pd.Timedelta(days=21)), ("commissioning", 0.1, due + pd.Timedelta(days=60))]
        else:
            ms = [("delivery", 1.0, due + pd.Timedelta(days=int(dict((s[0], s[5]) for s in SUPPLIERS)[it.supplier_id])))]
        for m, share, d in ms:
            rows.append({"po_id": po, "item_id": it.item_id, "supplier_id": it.supplier_id, "project_id": p, "qty": q, "due_date": due.normalize(),
                         "status": why, "milestone": m, "pay_date": max(d, DATA_DATE + pd.Timedelta(days=7)).normalize(), "currency": it.currency,
                         "amount_fx": round(amt_usd * share * fx, 2), "amount_usd_at_order": round(amt_usd * share, 2)})
    pd.DataFrame(rows).to_csv(OUT / "open_pos.csv", index=False)
    pd.DataFrame([{"currency": k, "per_usd_at_order": v} for k, v in FX_PER_USD.items()]).to_csv(OUT / "fx_at_order.csv", index=False)
    # supply network files
    fac = pd.DataFrame(FACTORIES, columns=["factory_id", "supplier_id", "factory", "city", "country", "lat", "lon", "items", "capacity_note"])
    fac.to_csv(OUT / "factories.csv", index=False)
    lanes = []
    for it in items[items.item_id != "CEM-OPC-B"].itertuples():
        segs = LANES[it.origin]
        rest = sum(m for _, m, _ in segs if m)
        rest_sd2 = sum(sd ** 2 for _, _, sd in segs if sd)
        mfg = max(2.0, it.lead_days - rest)
        mfg_sd = float(np.sqrt(max(1.0, it.lead_sd ** 2 - rest_sd2)))
        f = fac[fac["items"].str.split(";").apply(lambda x: it.item_id in x)].iloc[0]
        for k, (name, m, sd) in enumerate(segs):
            lanes.append({"item_id": it.item_id, "factory_id": f.factory_id, "seq": k + 1, "stage": name,
                          "mean_days": mfg if m is None else m, "sd_days": mfg_sd if sd is None else sd})
    lanes = pd.DataFrame(lanes)
    lanes.to_csv(OUT / "lanes.csv", index=False)
    pd.DataFrame(TIER2, columns=["item_id", "input", "tier2_supplier", "country", "single_source"]).to_csv(OUT / "tier2.csv", index=False)
    # where every open order is today: walk back from its due date through the lane
    ship = []
    rngs = np.random.default_rng(4242)
    opo = pd.DataFrame(rows).drop_duplicates("po_id")
    for r in opo.itertuples():
        segs = lanes[lanes.item_id == r.item_id].sort_values("seq")
        left = (pd.Timestamp(r.due_date) - DATA_DATE).days + float(rngs.normal(0, 3))
        cum, stage, into = 0.0, segs.stage.iloc[0], 0.0
        for sg in segs[::-1].itertuples():
            if left <= cum + sg.mean_days:
                stage, into = sg.stage, max(0.0, cum + sg.mean_days - left)
                break
            cum += sg.mean_days
        else:
            stage, into = "Not yet released to factory", 0.0
        ship.append({"po_id": r.po_id, "item_id": r.item_id, "project_id": r.project_id, "qty": r.qty, "status": r.status,
                     "current_stage": stage, "days_in_stage": round(into, 1), "promised_due": pd.Timestamp(r.due_date).normalize()})
    pd.DataFrame(ship).to_csv(OUT / "shipments.csv", index=False)

    print(f"items {len(items_out)}, ledger {len(led)}, deliveries {len(deliv)}, open PO lines {len(rows)}, plan rows {len(plan_rows)}")


if __name__ == "__main__":
    main()
