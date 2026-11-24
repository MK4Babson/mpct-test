"""
Build the synthetic global-markets dataset for the Global markets & trade page (data/macro/).

SYNTHETIC: shaped like real central-bank, balance-of-payments and labor data, with realistic orders of magnitude,
but not real figures. Every table can be replaced by an upload of the same columns (sources listed in the app).

  countries.csv         currency, central bank, policy framework (floor / corridor), reserve requirements, FX regime
  monetary.csv          monthly: monetary base, currency, checkable deposits, near money, excess reserves, rates,
                        inflation, expected inflation, real GDP growth, unemployment, FX per USD, REER
  bop.csv               annual balance of payments (USD bn) that satisfies CA + KA + NEO = FA + RRI, plus reserves,
                        short-term external debt and monthly imports
  labor_metro.csv       West Texas metros: unemployment, construction jobs, construction wages, electrician wage
  competing_projects.csv  large projects nearby that hire the same trades (fictional)
  tariffs.csv           tariff table: item, origin, legal basis, rate, dates, source and whether the rate is verified
  risk_seed.csv         likelihood ratings for the ten-driver supplier-country risk register

The money series are generated with the textbook's behaviour built in, so the sensitivities the app estimates have the
right signs: higher rates -> less cash per deposit (CC/D down), more time deposits (N/D up), fewer idle reserves (ER/D down).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "macro"
MONTHS = pd.period_range("2018-01", "2026-08", freq="M")

# code, name, currency, central bank, framework, rr on checkable, rr on near money, fx regime, capital mobility,
# broad money label, B0 (local bn), c, n, e, policy start, infl target, fx start (per USD), fx drift / yr, fx vol / mo
COUNTRIES = [
    ("US", "United States", "USD", "Federal Reserve", "floor (ample reserves)", 0.00, 0.00, "floating", "high", "M2",
     3500, 0.10, 0.20, 0.12, 1.5, 2.0, 1.0, 0.0, 0.0),
    ("IN", "India", "INR", "Reserve Bank of India", "corridor (repo, SDF floor)", 0.04, 0.04, "managed float", "medium", "M3",
     30000, 1.35, 7.2, 0.006, 6.0, 4.0, 70.0, 0.030, 0.008),
    ("KR", "Korea", "KRW", "Bank of Korea", "corridor", 0.07, 0.02, "floating", "high", "M2",
     230000, 0.12, 3.5, 0.004, 1.5, 2.0, 1100.0, 0.020, 0.012),
    ("MX", "Mexico", "MXN", "Banco de Mexico", "corridor", 0.00, 0.00, "floating", "high", "M2",
     1600, 0.55, 2.2, 0.010, 7.5, 3.0, 19.0, -0.005, 0.018),
    ("EA", "Euro area", "EUR", "European Central Bank", "floor (deposit facility)", 0.01, 0.01, "floating", "high", "M2",
     3300, 0.13, 0.85, 0.150, -0.4, 2.0, 0.85, 0.008, 0.015),
    ("JP", "Japan", "JPY", "Bank of Japan", "floor / yield control", 0.008, 0.008, "floating", "high", "M2",
     480000, 0.13, 0.75, 0.420, -0.1, 2.0, 110.0, 0.000, 0.012),
    ("CN", "China", "CNY", "People's Bank of China", "corridor + reserve ratio", 0.08, 0.08, "managed (band)", "low", "M2",
     32000, 0.12, 2.6, 0.020, 4.35, 3.0, 6.6, 0.010, 0.008),
]
CCOLS = ["code", "name", "currency", "central_bank", "framework", "rr_d", "rr_n", "fx_regime", "capital_mobility", "broad_label",
         "B0", "c0", "n0", "e0", "rate0", "target", "fx0", "fx_drift", "fx_vol"]
# policy-rate paths (start, key points): a hiking cycle in 2022-23 then cuts, country flavour
RATE_PATH = {
    "US": {"2018-01": 1.5, "2019-06": 2.4, "2020-03": 0.1, "2022-03": 0.3, "2023-07": 5.4, "2024-09": 5.0, "2025-12": 3.9, "2026-08": 3.6},
    "IN": {"2018-01": 6.0, "2019-10": 5.15, "2020-05": 4.0, "2022-05": 4.4, "2023-02": 6.5, "2025-02": 6.25, "2025-06": 5.5, "2026-08": 5.25},
    "KR": {"2018-01": 1.5, "2020-05": 0.5, "2021-08": 0.75, "2023-01": 3.5, "2024-10": 3.25, "2025-05": 2.5, "2026-08": 2.5},
    "MX": {"2018-01": 7.25, "2019-08": 8.0, "2021-02": 4.0, "2023-03": 11.25, "2024-03": 11.0, "2025-06": 8.0, "2026-08": 7.0},
    "EA": {"2018-01": -0.4, "2019-09": -0.5, "2022-07": 0.0, "2023-09": 4.0, "2024-06": 3.75, "2025-06": 2.0, "2026-08": 2.0},
    "JP": {"2018-01": -0.1, "2024-03": 0.1, "2024-07": 0.25, "2025-01": 0.5, "2026-08": 0.75},
    "CN": {"2018-01": 4.35, "2019-08": 4.25, "2020-04": 3.85, "2022-08": 3.65, "2023-08": 3.45, "2024-10": 3.1, "2025-05": 3.0, "2026-08": 3.0},
}


def _rng(s):
    return np.random.default_rng(sum(map(ord, s)) * 13 + 5)


def _path(points):
    s = pd.Series({pd.Period(k, "M"): v for k, v in points.items()}).reindex(MONTHS)
    return s.interpolate().bfill().ffill().values


def monetary():
    rows = []
    for c in COUNTRIES:
        r = dict(zip(CCOLS, c))
        rng = _rng(r["code"])
        rate = _path(RATE_PATH[r["code"]])
        n = len(MONTHS)
        # monetary base: trend growth, QE-style jumps in 2020, slower base when rates are high
        g = 0.006 + 0.004 * rng.standard_normal(n) - 0.0012 * np.clip(rate - rate.mean(), -3, 5)
        g[(MONTHS >= pd.Period("2020-03", "M")) & (MONTHS <= pd.Period("2021-06", "M"))] += 0.02 if r["code"] in ("US", "EA", "JP") else 0.006
        B = r["B0"] * np.exp(np.cumsum(g))
        d_rate = rate - rate[0]
        c_ = r["c0"] * np.exp(-0.035 * d_rate + 0.01 * np.cumsum(rng.standard_normal(n)) * 0.3)
        n_ = r["n0"] * np.exp(0.030 * d_rate + 0.01 * np.cumsum(rng.standard_normal(n)) * 0.3)
        e_ = r["e0"] * np.exp(-0.06 * d_rate + 0.02 * rng.standard_normal(n))
        if r["code"] in ("US", "EA", "JP"):
            e_ = e_ * np.where(MONTHS >= pd.Period("2020-03", "M"), 1.8, 1.0)
        den = c_ + r["rr_d"] + r["rr_n"] * n_ + e_
        D = B / den
        cc, nn, er = c_ * D, n_ * D, e_ * D
        lend = rate + {"US": 3.0, "IN": 3.5, "KR": 2.0, "MX": 4.0, "EA": 2.0, "JP": 0.9, "CN": 0.6}[r["code"]] + np.convolve(
            np.r_[np.full(5, d_rate[0]), d_rate], np.ones(6) / 6, "valid")[: n] * (0.6 - 1.0) + 0.05 * rng.standard_normal(n)
        infl = np.clip(r["target"] + np.r_[np.zeros(8), np.diff(rate, prepend=rate[0])[:-8]].cumsum() * 0 + 2.5 * np.sin(np.linspace(0, 3.2, n)) ** 2
                       * (MONTHS >= pd.Period("2021-03", "M")) * (MONTHS <= pd.Period("2024-06", "M")) + 0.3 * rng.standard_normal(n), -1.5, 12)
        exp_infl = 0.6 * r["target"] + 0.4 * pd.Series(infl).rolling(6, min_periods=1).mean().values
        gdp = np.clip(2.0 + {"IN": 4.0, "CN": 3.0, "KR": 0.5, "MX": -0.2, "EA": -0.6, "JP": -1.2, "US": 0.2}[r["code"]] - 0.25 * np.r_[np.zeros(9), d_rate[:-9]]
                      + 0.6 * rng.standard_normal(n), -12, 14)
        cov = (MONTHS >= pd.Period("2020-03", "M")) & (MONTHS <= pd.Period("2020-12", "M"))
        gdp[cov] -= 8
        unemp = np.clip({"US": 4.0, "IN": 7.5, "KR": 3.5, "MX": 3.3, "EA": 7.5, "JP": 2.6, "CN": 5.2}[r["code"]] - 0.15 * (gdp - gdp.mean())
                        + 0.1 * rng.standard_normal(n), 1.5, 15)
        fx = r["fx0"] * np.exp(np.cumsum(r["fx_drift"] / 12 + r["fx_vol"] * rng.standard_normal(n)))
        if r["code"] == "JP":
            fx *= np.where(MONTHS >= pd.Period("2022-03", "M"), np.linspace(1, 1.35, n), 1)
        if r["code"] == "US":
            fx = np.ones(n)
        price = np.cumprod(1 + infl / 1200)
        us_price = np.cumprod(1 + 2.6 / 1200 * np.ones(n))
        reer = 100 * (fx[0] / fx) * price / us_price
        res = {"US": 240, "IN": 400, "KR": 400, "MX": 175, "EA": 900, "JP": 1250, "CN": 3150}[r["code"]] * np.exp(np.cumsum(0.002 + 0.01 * rng.standard_normal(n)))
        for i, m in enumerate(MONTHS):
            rows.append({"country": r["code"], "month": str(m), "monetary_base": B[i], "currency": cc[i], "checkable_deposits": D[i],
                         "near_money": nn[i], "excess_reserves": er[i], "required_reserves": r["rr_d"] * D[i] + r["rr_n"] * nn[i],
                         "M1": cc[i] + D[i], "broad_money": cc[i] + D[i] + nn[i], "policy_rate": rate[i], "lending_rate": lend[i],
                         "cpi_yoy": infl[i], "expected_inflation": exp_infl[i], "real_gdp_yoy": gdp[i], "unemployment": unemp[i],
                         "fx_per_usd": fx[i], "reer": reer[i], "fx_reserves_usd_bn": res[i]})
    return pd.DataFrame(rows)


def bop():
    base = {  # CA, KA, NEO, RRI, exports, imports, short-term external debt, GDP (USD bn, illustrative scale)
        "US": (-850, -0.5, 20, 1, 2050, 3150, 1100, 28000), "IN": (-35, 0.1, -3, 40, 440, 690, 135, 3900),
        "KR": (60, -0.2, 2, 5, 680, 620, 160, 1750), "MX": (-15, 0.1, -8, 6, 610, 620, 60, 1850),
        "EA": (330, 5, -20, 10, 3100, 2800, 1400, 16500), "JP": (180, -3, 15, -20, 710, 760, 1300, 4100),
        "CN": (380, -0.2, -150, 15, 3500, 2600, 1150, 19000)}
    rows = []
    for k, (ca, ka, neo, rri, x, m, std, gdp) in base.items():
        rng = _rng(k + "bop")
        for i, y in enumerate(range(2019, 2026)):
            f = 1 + 0.04 * i + 0.05 * rng.standard_normal()
            ca_y, rri_y = ca * f + 0.15 * abs(ca) * rng.standard_normal(), rri + 0.5 * abs(rri + 5) * rng.standard_normal()
            neo_y = neo + 0.3 * abs(neo + 5) * rng.standard_normal()
            fa = ca_y + ka + neo_y - rri_y
            rows.append({"country": k, "year": y, "current_account": ca_y, "capital_account": ka, "net_errors_omissions": neo_y,
                         "financial_account": fa, "reserves_related_items": rri_y, "goods_exports": x * f, "goods_imports": m * f,
                         "short_term_external_debt": std * f, "gdp": gdp * (1 + 0.045 * i)})
    return pd.DataFrame(rows)


def labor():
    metros = {"Midland": (3.2, 10500, 33.5, 31.0), "Odessa": (3.9, 9800, 31.0, 29.5), "Lubbock": (3.4, 8900, 28.0, 27.0),
              "Abilene": (3.6, 5200, 26.5, 26.0)}
    rows = []
    for name, (u0, cj0, w0, el0) in metros.items():
        rng = _rng(name)
        n = len(MONTHS)
        oil = np.r_[np.zeros(26), -8 * np.exp(-np.arange(n - 26) / 6)]          # 2020 oil-price crash then recovery
        dc = np.clip(np.linspace(-1, 2.5, n), 0, None) * (name in ("Midland", "Odessa", "Lubbock"))   # data-center / power build-out
        u = np.clip(u0 - oil * 0.9 - 0.35 * dc + 0.15 * rng.standard_normal(n), 1.6, 14)
        cj = cj0 * (1 + 0.04 * oil / 8 + 0.06 * dc + 0.01 * rng.standard_normal(n))
        wage = w0 * np.cumprod(1 + (0.035 + 0.012 * dc) / 12 + 0.002 * rng.standard_normal(n))
        el = el0 * np.cumprod(1 + (0.038 + 0.018 * dc) / 12 + 0.002 * rng.standard_normal(n))
        for i, m in enumerate(MONTHS):
            rows.append({"metro": name, "month": str(m), "unemployment": u[i], "construction_jobs": cj[i], "construction_avg_hourly": wage[i],
                         "electrician_hourly": el[i]})
    return pd.DataFrame(rows)


COMPETING = [
    ("Caprock AI campus (fictional)", "Data center", "Lubbock", "2026-06", "2027-05", 520),
    ("Permian gas plant expansion (fictional)", "Oil & gas", "Midland", "2026-03", "2027-01", 260),
    ("Llano Estacado substation + 345 kV line (fictional)", "Power / grid", "Lubbock", "2026-08", "2027-04", 180),
    ("Basin hyperscale phase 2 (fictional)", "Data center", "Odessa", "2026-10", "2027-09", 450),
    ("Big Spring battery storage (fictional)", "Power / storage", "Midland", "2026-05", "2026-12", 90),
]
TRADE_SUPPLY = {"Midland": 1400, "Odessa": 1100, "Lubbock": 1250, "Abilene": 600}   # estimated licensed electricians available (synthetic)

# item_id, origin, legal basis, rate %, effective from, effective to, applies to, status, source
SRC_232 = "https://www.cov.com/en/news-and-insights/insights/2026/04/current-and-forthcoming-section-232-actions-by-the-trump-administration"
SRC_122 = "https://www.wilmerhale.com/en/sitecore/content/whlaunch/home/research/news-publications/20260220-supreme-court-strikes-down-ieepa-tariffs-what-now"
SRC_WH = "https://www.whitehouse.gov/fact-sheets/2026/04/fact-sheet-president-donald-j-trump-strengthens-tariffs-on-steel-aluminum-and-copper-imports"
TARIFFS = [
    ("PIPE-CHW", "India", "Section 232 (steel article)", 50.0, "2026-04-06", "", "full customs value", "rate from source; confirm HTS classification", SRC_232),
    ("RBR-5", "*non-US*", "Section 232 (steel article)", 50.0, "2026-04-06", "", "full customs value", "rate from source; confirm HTS classification", SRC_232),
    ("STL-W", "*non-US*", "Section 232 (steel article)", 50.0, "2026-04-06", "", "full customs value", "rate from source; confirm HTS classification", SRC_232),
    ("MV-500", "Korea", "Section 232 (copper derivative)", 25.0, "2026-04-06", "", "full customs value",
     "ILLUSTRATIVE: derivatives 25-50%; Korea deal may lower it; verify HTS", SRC_232),
    ("XFMR", "Korea", "Section 232 (grid equipment, reduced rate)", 15.0, "2026-04-06", "2027-12-31", "full customs value",
     "category rate from source (through 2027); confirm the item is in scope", SRC_232),
    ("SWG", "Mexico", "Section 232 (grid equipment, reduced rate)", 15.0, "2026-04-06", "2027-12-31", "full customs value",
     "category rate from source; USMCA US-content portion may enter at 0%; verify", SRC_232),
    ("BUS-4K", "Mexico", "Section 232 (grid equipment, reduced rate)", 15.0, "2026-04-06", "2027-12-31", "full customs value",
     "ILLUSTRATIVE: confirm busway is in the grid-equipment scope", SRC_232),
    ("CRAH", "China", "Section 301 (China)", 25.0, "2018-09-24", "", "customs value", "ILLUSTRATIVE: Section 301 list rate; verify HTS", SRC_232),
    ("CRAH", "China", "MFN (normal trade relations)", 1.4, "1995-01-01", "", "customs value", "ILLUSTRATIVE MFN rate; verify HTS", ""),
    ("CHL", "Germany", "MFN (normal trade relations)", 1.9, "1995-01-01", "", "customs value", "ILLUSTRATIVE MFN rate; verify HTS", ""),
    ("CDU", "Japan", "MFN (normal trade relations)", 2.0, "1995-01-01", "", "customs value", "ILLUSTRATIVE MFN rate; verify HTS", ""),
    ("UPS", "India", "MFN (normal trade relations)", 1.5, "1995-01-01", "", "customs value", "ILLUSTRATIVE MFN rate; verify HTS", ""),
    ("*", "*non-US*", "Section 122 (temporary global 10%)", 10.0, "2026-02-24", "2026-07-23", "customs value; not on Section 232 goods",
     "from source: 150-day window; check whether it was extended", SRC_122),
]

DRIVERS = ["Labor & demographics", "Energy & raw materials", "Export policy (origin country)", "Import policy (US tariffs)",
           "Currency & macro", "Logistics chokepoints", "Natural hazards & climate", "Geopolitics & sanctions",
           "Concentration (single source)", "Quality & compliance"]
# likelihood 1-3 by origin and driver (synthetic judgement, editable in the app)
RISK_SEED = {
    "USA": [2, 1, 1, 1, 1, 1, 2, 1, 2, 1], "Mexico": [1, 2, 1, 2, 2, 1, 1, 1, 2, 1], "India": [1, 2, 3, 3, 2, 2, 2, 1, 2, 2],
    "Korea": [2, 1, 1, 2, 2, 2, 1, 1, 3, 1], "Germany": [2, 2, 1, 1, 2, 2, 1, 1, 2, 1], "China": [1, 1, 3, 3, 2, 2, 1, 3, 2, 2],
    "Japan": [3, 2, 1, 1, 2, 2, 2, 1, 2, 1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(COUNTRIES, columns=CCOLS).to_csv(OUT / "countries.csv", index=False)
    monetary().to_csv(OUT / "monetary.csv", index=False)
    bop().to_csv(OUT / "bop.csv", index=False)
    labor().to_csv(OUT / "labor_metro.csv", index=False)
    cp = pd.DataFrame(COMPETING, columns=["project", "type", "metro", "start", "finish", "electricians_peak"])
    cp.to_csv(OUT / "competing_projects.csv", index=False)
    pd.DataFrame([{"metro": k, "electricians_available": v} for k, v in TRADE_SUPPLY.items()]).to_csv(OUT / "trade_supply.csv", index=False)
    pd.DataFrame(TARIFFS, columns=["item_id", "origin", "legal_basis", "rate_pct", "effective_from", "effective_to", "applies_to", "status", "source"]).to_csv(
        OUT / "tariffs.csv", index=False)
    pd.DataFrame([{"origin": o, "driver": d, "likelihood": l} for o, ls in RISK_SEED.items() for d, l in zip(DRIVERS, ls)]).to_csv(
        OUT / "risk_seed.csv", index=False)
    print("macro data written to", OUT)


if __name__ == "__main__":
    main()
