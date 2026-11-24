"""
Shared configuration for the Multi-Prime Commercial Control Tower.

ALL DATA IS SYNTHETIC. Contractor names, sites, quantities and rates are
illustrative and do not represent any real company, project or contract.
"""

import os
SEED = int(os.environ.get("MPCT_SEED", 42))  # change the seed for a hold-out test

# Billing periods (monthly pay applications)
PERIODS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08"]

# Six fictional buildings across three fictional West Texas campuses, starting at different
# times so the portfolio shows projects at different stages.
# scale = size of scope vs the base rate card; start = index of the first billing month.
PROJECTS = [
    {"project_id": "WTX-A1", "campus": "Mesa Campus (fictional)",   "scale": 1.00, "start": 0},
    {"project_id": "WTX-A2", "campus": "Mesa Campus (fictional)",   "scale": 0.85, "start": 0},
    {"project_id": "WTX-B1", "campus": "Ridge Campus (fictional)",  "scale": 1.15, "start": 1},
    {"project_id": "WTX-B2", "campus": "Ridge Campus (fictional)",  "scale": 0.90, "start": 2},
    {"project_id": "WTX-C1", "campus": "Plains Campus (fictional)", "scale": 1.05, "start": 3},
    {"project_id": "WTX-C2", "campus": "Plains Campus (fictional)", "scale": 0.80, "start": 4},
]

# Three prime contractors under a multi-prime delivery model (fictional).
PRIMES = [
    {"prime_id": "PRM-E", "prime_name": "Lone Mesa Electrical Constructors (fictional)", "trade": "Electrical"},
    {"prime_id": "PRM-M", "prime_name": "Red Draw Mechanical Services (fictional)",      "trade": "Mechanical"},
    {"prime_id": "PRM-C", "prime_name": "Brazos Plains Civil Works (fictional)",         "trade": "Civil"},
]

# Contract rate cards = schedule of values (SOV) per building at scale 1.0.
# (item_code, description, unit, base_qty, contract_unit_rate, start_month_idx, duration_months)
# Lump-sum (LS) lines are billed by % complete; base_qty = 1 and rate = scheduled value.
RATE_CARDS = {
    "PRM-E": [
        ("E-101", "Cable tray 24in ladder, installed",           "LF",  18000,    62.00, 1, 5),
        ("E-102", "MV cable 15kV 500kcmil, installed",           "LF",  42000,    48.50, 2, 4),
        ("E-103", "Conduit 4in PVC-coated RGS, installed",       "LF",  26000,    71.00, 1, 5),
        ("E-104", "Busway 4000A, installed",                     "LF",   3200,  1150.00, 2, 4),
        ("E-105", "Switchgear lineup - set and terminate",       "EA",     24, 38500.00, 2, 4),
        ("E-106", "UPS module - set and terminate",              "EA",     48, 21000.00, 3, 3),
        ("E-107", "Generator set - place and terminate",         "EA",     36, 64000.00, 2, 4),
        ("E-LS1", "Mobilization",                                "LS",      1, 850000.00, 0, 2),
        ("E-LS2", "Commissioning support (L3-L5)",               "LS",      1, 1200000.00, 4, 3),
    ],
    "PRM-M": [
        ("M-201", "Chilled water pipe 12in welded steel",        "LF",  14000,   310.00, 1, 5),
        ("M-202", "Process/condenser pipe 8in",                  "LF",  22000,   185.00, 1, 5),
        ("M-203", "Coolant distribution unit (CDU) - set/connect","EA",   120, 14500.00, 3, 3),
        ("M-204", "Air-cooled chiller - set/connect",            "EA",     40, 52000.00, 2, 4),
        ("M-205", "CRAH unit - set/connect",                     "EA",    160,  9800.00, 2, 4),
        ("M-206", "Galvanized ductwork",                         "LB", 260000,    14.75, 1, 5),
        ("M-LS1", "Mobilization",                                "LS",      1, 650000.00, 0, 2),
        ("M-LS2", "TAB and commissioning support",               "LS",      1, 900000.00, 4, 3),
    ],
    "PRM-C": [
        ("C-301", "Mass excavation",                             "CY", 420000,     9.40, 0, 3),
        ("C-302", "Structural concrete - foundations",           "CY",  38000,   465.00, 0, 4),
        ("C-303", "Reinforcing steel",                           "TON",  5200,  2350.00, 0, 4),
        ("C-304", "Aggregate base course",                       "SY", 180000,    18.50, 1, 4),
        ("C-305", "Storm drain pipe 36in RCP",                   "LF",   9500,   165.00, 1, 3),
        ("C-306", "Concrete-encased duct bank",                  "LF",  12000,   240.00, 1, 4),
        ("C-LS1", "Mobilization",                                "LS",      1, 500000.00, 0, 2),
        ("C-LS2", "SWPPP and erosion control",                   "LS",      1, 300000.00, 0, 6),
    ],
}

# Time-and-materials labor rates (per hour) by prime
TM_RATES = {
    "PRM-E": [("Journeyman electrician", 98.00), ("Electrical foreman", 112.00)],
    "PRM-M": [("Pipefitter", 96.00), ("Sheet metal worker", 91.00)],
    "PRM-C": [("Equipment operator", 88.00), ("Laborer", 62.00)],
}

# Base field headcount per building at scale 1.0 (drives per diem lines)
BASE_HEADCOUNT = {"PRM-E": 180, "PRM-M": 140, "PRM-C": 110}
WORKDAYS_PER_MONTH = 22

# Commercial policy (illustrative values; in a real program these come from the contract)
RETAINAGE_RATE = 0.10          # withheld on SOV + change-order work, not on T&M or per diem
PER_DIEM_CAP = 178.00          # $/worker-day (illustrative: lodging + meals cap)

# Governance tolerances used by the engine (defaults; evaluate.py sweeps these)
TOLERANCES = {
    "qty_review_pct": 1.5,      # billed qty above verified qty by more than this % -> REVIEW
    "qty_hold_pct": 5.0,        # ... by more than this % -> HOLD
    "rate_round_usd": 0.005,    # unit rate above contract by more than half a cent -> HOLD (rounding only:
                                #   a contract rate is a legal number, so there is no % tolerance on rates)
    "pct_review_pts": 1.0,      # LS % complete billed above verified by more than N points -> REVIEW
    "pct_hold_pts": 5.0,        # ... more than N points -> HOLD
    "tm_review_pct": 0.0,       # T&M hours above pre-approved -> REVIEW
    "tm_hold_pct": 10.0,        # ... by more than this % -> HOLD
    "money_tol": 1.00,          # $ rounding tolerance for arithmetic checks
}
