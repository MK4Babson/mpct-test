"""
Build the reference datasets used by the cost-intelligence, process and pattern features.

ALL OF THESE ARE SYNTHETIC unless you replace them. Each file says so in a `source` column or in the README.
  data/market/indices_synthetic.csv     monthly price indices 2018-01..2026-10 (steel, copper, concrete, labor, equipment)
  data/market/history_monthly.csv       60 past packages: monthly cost growth vs their trade's index (for beta / alpha)
  data/market/rate_history.csv          1,600 past bid rates with attributes (for the hedonic should-cost model)
  data/market/building_history.csv      14 past repeated data center buildings (for the learning curve)
  data/market/portfolio_cash_history.csv owner's monthly capital spend 2018-01..2026-10 (for the seasonal cash forecast)
  data/market/carbon_factors.csv        approximate cradle-to-gate carbon factors per item (replace with EPDs)
  data/training/co_text.csv             600 labelled change-order justifications (for the Naive Bayes text model)
  <program>/co_details.csv              justification text, cause and schedule impact for each change order
  <program>/approval_log.csv            stage timestamps for each bill (for cycle time and Little's Law)

Usage:  python src/build_extras.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RATE_CARDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MARKET = ROOT / "data" / "market"
PROGRAMS = {"raw": ROOT / "data" / "raw", "north_koel": ROOT / "data" / "north_koel", "clean_demo": ROOT / "data" / "clean_demo"}
MONTHS = pd.period_range("2018-01", "2026-10", freq="M")

# Which price index drives each trade's costs (weights), and the share of each item's rate that is material
TRADE_BASKET = {"Electrical": {"copper": 0.45, "steel": 0.15, "equipment": 0.40},
                "Mechanical": {"steel": 0.45, "equipment": 0.55},
                "Civil": {"concrete": 0.55, "steel": 0.45}}
FAMILY = {  # item_code -> (family, trade, index, material share)
    "E-101": ("Cable tray", "Electrical", "steel", 0.55), "E-102": ("MV cable", "Electrical", "copper", 0.65),
    "E-103": ("Conduit", "Electrical", "steel", 0.50), "E-104": ("Busway", "Electrical", "copper", 0.70),
    "E-105": ("Switchgear set/terminate", "Electrical", "equipment", 0.15), "E-106": ("UPS set/terminate", "Electrical", "equipment", 0.15),
    "E-107": ("Generator set/terminate", "Electrical", "equipment", 0.15),
    "M-201": ("Chilled water pipe 12in", "Mechanical", "steel", 0.50), "M-202": ("Process pipe 8in", "Mechanical", "steel", 0.45),
    "M-203": ("CDU set/connect", "Mechanical", "equipment", 0.15), "M-204": ("Chiller set/connect", "Mechanical", "equipment", 0.15),
    "M-205": ("CRAH set/connect", "Mechanical", "equipment", 0.15), "M-206": ("Galvanized ductwork", "Mechanical", "steel", 0.55),
    "C-301": ("Mass excavation", "Civil", "labor", 0.05), "C-302": ("Structural concrete", "Civil", "concrete", 0.55),
    "C-303": ("Reinforcing steel", "Civil", "steel", 0.65), "C-304": ("Aggregate base", "Civil", "concrete", 0.45),
    "C-305": ("Storm drain RCP 36in", "Civil", "concrete", 0.50), "C-306": ("Duct bank", "Civil", "concrete", 0.45),
}
REGIONS = {"West Texas": 0.0, "Gulf Coast": 0.05, "Midwest": 0.03, "Southeast": -0.04, "Mountain West": 0.06}


def indices(rng):
    t = np.arange(len(MONTHS))
    yr = t / 12

    def bump(center, width, height):
        return height * np.exp(-0.5 * ((t - center) / width) ** 2)

    i2021 = MONTHS.get_loc(pd.Period("2021-09", "M"))
    i2025 = MONTHS.get_loc(pd.Period("2025-06", "M"))
    noise = lambda sd: np.cumsum(rng.normal(0, sd, len(t))) * 0.3  # noqa: E731
    df = pd.DataFrame({
        "period": MONTHS.astype(str),
        "steel": 100 * np.exp(0.015 * yr) * (1 + bump(i2021, 7, 0.85) + 0.18 / (1 + np.exp(-(t - i2025) / 2))) * (1 + noise(0.012)),
        "copper": 100 * np.exp(0.055 * yr) * (1 + bump(i2021 + 4, 9, 0.25)) * (1 + noise(0.02)),
        "concrete": 100 * np.exp(0.045 * yr) * (1 + 0.10 / (1 + np.exp(-(t - i2021 - 6) / 3))) * (1 + noise(0.005)),
        "labor": 100 * np.exp(0.038 * yr) * (1 + noise(0.003)),
        "equipment": 100 * np.exp(0.035 * yr) * (1 + 0.16 / (1 + np.exp(-(t - i2021 - 8) / 4))) * (1 + noise(0.006)),
    })
    for c in ["steel", "copper", "concrete", "labor", "equipment"]:
        df[c] = (100 * df[c] / df[c].iloc[0]).round(2)
    df["source"] = "synthetic (replace with BLS / FRED series)"
    return df


def trade_returns(idx):
    r = idx.set_index("period")[["steel", "copper", "concrete", "labor", "equipment"]].pct_change() * 100
    return pd.DataFrame({tr: sum(w * r[k] for k, w in b.items()) for tr, b in TRADE_BASKET.items()})


def history_monthly(rng, idx):
    rets = trade_returns(idx)
    true_beta = {"Electrical": 0.42, "Mechanical": 0.38, "Civil": 0.33}
    rows, pk = [], []
    for i in range(60):
        trade = ["Electrical", "Mechanical", "Civil"][i % 3]
        start = int(rng.integers(1, len(MONTHS) - 30))
        dur = int(rng.integers(12, 25))
        tier = rng.choice(["Tier 1", "Tier 2", "Tier 3"], p=[0.4, 0.4, 0.2])
        alpha = {"Tier 1": 0.02, "Tier 2": 0.08, "Tier 3": 0.18}[tier] + rng.normal(0, 0.03)
        pid = f"HIST-{trade[0]}{i + 1:02d}"
        for k in range(dur):
            per = MONTHS[start + k]
            x = rets[trade].iloc[start + k]
            y = alpha + true_beta[trade] * x + rng.normal(0, 0.30)
            rows.append({"package_id": pid, "trade": trade, "contractor_tier": tier, "period": str(per),
                         "index_change_pct": round(x, 4), "cost_growth_pct": round(y, 4)})
        pk.append(pid)
    return pd.DataFrame(rows)


def rate_history(rng, idx):
    ix = idx.set_index("period")
    ref = pd.Period("2026-01", "M")
    base = {code: rate for cards in RATE_CARDS.values() for code, _, unit, qty, rate, *_ in cards if unit != "LS"}
    ref_qty = {code: qty for cards in RATE_CARDS.values() for code, _, unit, qty, rate, *_ in cards if unit != "LS"}
    unit_of = {code: unit for cards in RATE_CARDS.values() for code, _, unit, *_ in cards}
    rows = []
    for _ in range(1600):
        code = rng.choice(list(FAMILY))
        fam, trade, index, ms = FAMILY[code]
        per = MONTHS[int(rng.integers(0, len(MONTHS) - 1))]
        region = rng.choice(list(REGIONS), p=[0.3, 0.2, 0.2, 0.15, 0.15])
        union = int(rng.random() < 0.3)
        accel = int(rng.random() < 0.25)
        diff = int(rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1]))
        qty = ref_qty[code] * float(np.exp(rng.normal(0, 0.6)))
        idx_ratio = ix.loc[str(per), index] / ix.loc[str(ref), index]
        lab_ratio = ix.loc[str(per), "labor"] / ix.loc[str(ref), "labor"]
        log_rate = (np.log(base[code]) + ms * np.log(idx_ratio) + (1 - ms) * np.log(lab_ratio)
                    - 0.07 * np.log(qty / ref_qty[code]) + REGIONS[region] + 0.12 * union + 0.08 * accel
                    + 0.06 * (diff - 1) + rng.normal(0, 0.07))
        rows.append({"bid_id": f"BID-{len(rows) + 1:05d}", "item_code": code, "family": fam, "trade": trade, "unit": unit_of[code],
                     "period": str(per), "region": region, "union_labor": union, "accelerated": accel,
                     "site_difficulty": diff, "quantity": round(qty, 1), "unit_rate": round(float(np.exp(log_rate)), 2)})
    return pd.DataFrame(rows)


def building_history(rng):
    """14 past data-center buildings by the same owner: cost per MW falls with repetition (true learning rate 93%)."""
    b = np.log2(0.93)
    rows = []
    for n in range(1, 15):
        mw = float(rng.choice([24, 32, 36, 40, 48]))
        cost_per_mw = 11.2e6 * n ** b * float(np.exp(rng.normal(0, 0.035)))
        months = 15.5 * n ** np.log2(0.95) * float(np.exp(rng.normal(0, 0.04)))
        rows.append({"building": f"Past building {n}", "sequence": n, "year": 2018 + (n - 1) // 2, "capacity_mw": mw,
                     "cost_per_mw": round(cost_per_mw, 0), "build_months": round(months, 1), "source": "synthetic"})
    return pd.DataFrame(rows)


def cash_history(rng):
    t = np.arange(len(MONTHS))
    season = {1: 0.78, 2: 0.82, 3: 0.95, 4: 1.0, 5: 1.05, 6: 1.10, 7: 1.08, 8: 1.06, 9: 1.04, 10: 1.0, 11: 0.96, 12: 1.16}
    spend = [42e6 * np.exp(0.012 * k) * season[p.month] * float(np.exp(rng.normal(0, 0.045))) for k, p in zip(t, MONTHS)]
    return pd.DataFrame({"period": MONTHS.astype(str), "capital_spend": np.round(spend, 0), "source": "synthetic"})


# ---------------- change-order text ----------------
CAUSES = {
    "Owner design change": (True, ["Owner directed revision per RFI-{n} and revised drawing {d}, rev {r}.",
                                   "Design change issued by the owner's engineer: added {w} per bulletin {n}.",
                                   "Scope added at owner request to support increased rack density; see ASI-{n}."]),
    "Unforeseen site condition": (True, ["Unforeseen subsurface rock encountered at {loc}; geotech report {n} attached.",
                                         "Differing site condition: groundwater at {loc} not shown in the geotechnical baseline.",
                                         "Buried utility not shown on survey found at {loc}; relocation required."]),
    "Code / authority requirement": (True, ["Authority having jurisdiction required {w} at inspection {n}.",
                                            "Utility interconnection requirement changed; {w} mandated by the utility.",
                                            "Fire marshal comment {n} requires {w}."]),
    "Already in base scope": (False, ["Additional {w} needed to complete the system as installed.",
                                      "Scope clarification: {w} required for a complete installation.",
                                      "Extra {w} for coordination with other trades."]),
    "Contractor productivity / rework": (False, ["Additional labor required due to rework of {w} at {loc}.",
                                                  "Crew productivity lower than planned; extra hours for {w}.",
                                                  "Re-installation of {w} after failed inspection."]),
    "Acceleration not directed": (False, ["Overtime premium to recover schedule on {w}.",
                                          "Added second shift to maintain milestone dates for {w}.",
                                          "Acceleration costs for {w} to meet our recovery schedule."]),
}
WORDS = ["cable tray supports", "grounding conductors", "pipe hangers", "duct bank encasement", "chiller pads",
         "fire stopping", "seismic bracing", "conduit runs", "condenser water branch", "rebar dowels"]
LOCS = ["grid line C-7", "the north utility yard", "electrical room 2", "the MEP corridor", "column line 14", "the switchyard"]


def co_text(rng, cause):
    _, tmpl = CAUSES[cause]
    s = rng.choice(tmpl)
    return s.format(n=int(rng.integers(100, 999)), d=f"E-{int(rng.integers(100, 699))}", r=int(rng.integers(1, 6)),
                    w=rng.choice(WORDS), loc=rng.choice(LOCS))


def co_training(rng):
    rows = []
    causes = list(CAUSES)
    for _ in range(600):
        c = rng.choice(causes)
        txt = co_text(rng, c)
        if rng.random() < 0.15:  # some messy wording, so the model is not trivially perfect
            txt += " " + co_text(rng, rng.choice(causes)).split(" ", 3)[-1]
        rows.append({"text": txt, "cause": c, "justified": int(CAUSES[c][0])})
    df = pd.DataFrame(rows)
    flip = rng.random(len(df)) < 0.06  # label noise: real reviewers disagree sometimes
    df.loc[flip, "justified"] = 1 - df.loc[flip, "justified"]
    df["source"] = "synthetic"
    return df


def co_details(rng, folder):
    co = pd.read_csv(folder / "change_orders.csv")
    p = [0.28, 0.18, 0.12, 0.17, 0.13, 0.12]
    rows = []
    for r in co.itertuples():
        cause = rng.choice(list(CAUSES), p=p)
        txt = co_text(rng, cause)
        if "description" in co.columns and isinstance(r.description, str) and not r.description.startswith("Added scope"):
            txt = r.description + ". " + txt
        rows.append({"co_id": r.co_id, "justification": txt, "cause_category": cause, "raised_by": rng.choice(["Contractor", "Owner", "Designer"], p=[0.6, 0.25, 0.15]) if CAUSES[cause][0] else "Contractor",
                     "schedule_days_if_rejected": int(rng.choice([0, 0, 5, 10, 15, 20, 30])),
                     "source": "synthetic"})
    pd.DataFrame(rows).to_csv(folder / "co_details.csv", index=False)


def approval_log(rng, folder):
    from run_engine import run
    from schema import coerce, read_folder
    tables = {k: coerce(v, k) for k, v in read_folder(folder).items()}
    _, gates = run(tables=tables)
    pay = pd.read_csv(folder / "payments.csv", parse_dates=["submitted_date", "paid_date"])
    pay = pay.merge(gates[["invoice_id", "gate"]], on="invoice_id", how="left")
    stages = ["Auto-match (rules engine)", "Field verification", "PM approval", "Finance / AP"]
    weights = np.array([0.03, 0.45, 0.22, 0.30])
    data_date = pay.submitted_date.max() + pd.Timedelta(days=5)
    rows = []
    for r in pay.itertuples():
        start = r.submitted_date
        total = (r.paid_date - start).days if pd.notna(r.paid_date) else None
        st_list = stages + (["Dispute resolution"] if r.gate == "HOLD" else [])
        w = rng.dirichlet(weights * 20) if r.gate != "HOLD" else np.append(rng.dirichlet(weights * 20) * 0.55, 0.45)
        if total is None:  # still open: in progress at the data date
            total = max(1, (data_date - start).days + int(rng.integers(3, 15)))
        t = start
        for s, f in zip(st_list, w):
            d = max(0.2, total * f)
            end = t + pd.Timedelta(days=d)
            done = end <= data_date and (pd.notna(r.paid_date) or end <= data_date - pd.Timedelta(days=1))
            rows.append({"invoice_id": r.invoice_id, "stage": s, "start": t.strftime("%Y-%m-%d %H:%M"),
                         "end": end.strftime("%Y-%m-%d %H:%M") if done else "", "source": "synthetic"})
            if not done:
                break
            t = end
    pd.DataFrame(rows).to_csv(folder / "approval_log.csv", index=False)


CARBON = [  # item_code, unit, kg of material per unit, factor kgCO2e/kg, material, note
    ("E-101", "LF", 4.5, 1.55, "Steel (galvanized tray)", "~4.5 kg/ft for 24in ladder tray; ICE-style steel section factor"),
    ("E-102", "LF", 2.2, 3.81, "Copper + insulation", "~2.2 kg/ft for 500kcmil Cu cable; copper ~3.8 kgCO2e/kg (primary-heavy mix)"),
    ("E-103", "LF", 4.9, 1.55, "Steel (RGS conduit)", "~4.9 kg/ft for 4in rigid steel conduit"),
    ("E-104", "LF", 45.0, 3.0, "Copper/aluminium busbar", "~45 kg/ft for 4000A busway; blended Cu/Al factor"),
    ("C-302", "CY", 1835.0, 0.13, "Concrete (RC 32/40)", "2,400 kg/m3 x 0.7646 m3/CY; ICE-style ~0.13 kgCO2e/kg"),
    ("C-303", "TON", 907.2, 1.99, "Reinforcing steel", "short ton; ICE v3-style rebar ~1.99 kgCO2e/kg (world average)"),
    ("C-304", "SY", 150.0, 0.0052, "Crushed aggregate", "~150 kg/SY at 6in; aggregate ~0.005 kgCO2e/kg"),
    ("C-305", "LF", 360.0, 0.13, "Precast concrete pipe", "~360 kg/ft for 36in RCP"),
    ("C-306", "LF", 600.0, 0.13, "Concrete encasement", "~0.33 m3/ft encasement"),
    ("M-201", "LF", 74.0, 1.55, "Steel pipe", "12in Sch 40 ~74 kg/ft"),
    ("M-202", "LF", 43.0, 1.55, "Steel pipe", "8in Sch 40 ~43 kg/ft"),
    ("M-206", "LB", 0.4536, 2.76, "Galvanized sheet steel", "per lb of ductwork; galvanized sheet ~2.8 kgCO2e/kg"),
]
SPEND_BASED = 0.30  # kgCO2e per $ for equipment set/terminate lines (rough spend-based factor for electrical/HVAC equipment)


def carbon_factors():
    df = pd.DataFrame(CARBON, columns=["item_code", "unit", "kg_per_unit", "kgco2e_per_kg", "material", "basis"])
    df["kgco2e_per_unit"] = (df.kg_per_unit * df.kgco2e_per_kg).round(3)
    df["source"] = "approximate generic cradle-to-gate (A1-A3) factors in the style of the ICE database; replace with product EPDs"
    return df


def main():
    rng = np.random.default_rng(20261119)
    MARKET.mkdir(parents=True, exist_ok=True)
    idx = indices(rng)
    idx.to_csv(MARKET / "indices_synthetic.csv", index=False)
    history_monthly(rng, idx).to_csv(MARKET / "history_monthly.csv", index=False)
    rate_history(rng, idx).to_csv(MARKET / "rate_history.csv", index=False)
    building_history(rng).to_csv(MARKET / "building_history.csv", index=False)
    cash_history(rng).to_csv(MARKET / "portfolio_cash_history.csv", index=False)
    carbon_factors().to_csv(MARKET / "carbon_factors.csv", index=False)
    (ROOT / "data" / "training").mkdir(exist_ok=True)
    co_training(rng).to_csv(ROOT / "data" / "training" / "co_text.csv", index=False)
    for name, folder in PROGRAMS.items():
        co_details(np.random.default_rng(sum(map(ord, name))), folder)
        approval_log(np.random.default_rng(7 + len(name)), folder)
    print("wrote", sorted(p.name for p in MARKET.iterdir()))


if __name__ == "__main__":
    main()
