"""
Patterns in billing behaviour.

  F12  control charts   p-chart of the share of bill lines with an issue, per contractor per month (3-sigma limits, run rules)
  F13  segments         k-means on contract packages (one contractor on one project), mapped to a review policy
  F14  issue patterns   association rules between issue types on the same bill (support, confidence, lift)
  F15  lead-lag tests   Granger tests on the pooled 13-program training history: do pending change orders precede errors?
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training"


# ---------------------------------------------------------------- F12
def p_chart(tables, gates, exceptions, names=None):
    lines = tables["invoice_lines"].merge(gates[["invoice_id", "prime_id", "period"]], on="invoice_id")
    bad_lines = exceptions.dropna(subset=["line_no"])[["invoice_id", "line_no"]].drop_duplicates()
    bad_lines["bad"] = 1
    lines = lines.merge(bad_lines, on=["invoice_id", "line_no"], how="left")
    # bill-level issues (duplicates, retainage, header) mark every line of that bill? no: count them as one bad line
    hdr = exceptions[exceptions.line_no.isna()].groupby("invoice_id").size().rename("hdr")
    g = lines.groupby(["prime_id", "period"]).agg(n=("line_no", "size"), x=("bad", "sum"))
    hb = gates.set_index("invoice_id").join(hdr).groupby(["prime_id", "period"]).hdr.sum()
    g["x"] = g.x.fillna(0) + hb.reindex(g.index).fillna(0)
    g = g.reset_index()
    out = []
    for prime, d in g.groupby("prime_id"):
        d = d.sort_values("period").copy()
        pbar = d.x.sum() / d.n.sum()
        d["p"] = d.x / d.n
        d["center"] = pbar
        d["ucl"] = (pbar + 3 * np.sqrt(pbar * (1 - pbar) / d.n)).clip(upper=1)
        d["lcl"] = (pbar - 3 * np.sqrt(pbar * (1 - pbar) / d.n)).clip(lower=0)
        d["out_of_control"] = d.p > d.ucl
        above = (d.p > pbar).astype(int).values
        run = np.zeros(len(d), int)
        for i in range(len(d)):
            run[i] = run[i - 1] + 1 if (i and above[i]) else above[i]
        d["run_above"] = run
        # rule: 6 increasing points in a row = trend
        inc = np.r_[0, np.diff(d.p.values) > 0].astype(int)
        trend = np.zeros(len(d), int)
        for i in range(len(d)):
            trend[i] = trend[i - 1] + 1 if (i and inc[i]) else 0
        d["rising_run"] = trend
        d["contractor"] = (names or {}).get(prime, prime)
        out.append(d)
    pc = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    verdict = []
    for prime, d in pc.groupby("prime_id"):
        last = d.iloc[-1]
        sig = []
        if d.out_of_control.any():
            sig.append(f"{int(d.out_of_control.sum())} month(s) above the upper limit ({', '.join(d[d.out_of_control].period)})")
        if d.run_above.max() >= 7:
            sig.append("7+ months in a row above average")
        if d.rising_run.max() >= 5:
            sig.append("5+ rising months in a row")
        verdict.append({"prime_id": prime, "contractor": last.contractor, "average_issue_rate": last.center, "latest": last.p,
                        "signal": "Getting worse (special cause)" if sig else "Stable: normal variation",
                        "evidence": "; ".join(sig) if sig else "every month inside the 3-sigma limits"})
    return pc, pd.DataFrame(verdict)


# ---------------------------------------------------------------- F13
SEG_FEATURES = {"first_pass_pct": "First-pass rate (%)", "risk_pct": "Money at risk (% of billed)",
                "spi": "Schedule index", "co_share_pct": "Change orders (% of contract)",
                "tm_share_pct": "T&M (% of billed)", "avg_jump": "Average bill jump (log)"}
POLICY = {0: ("Light touch", "Sample 1 in 4 bills; auto-pay PASS bills"),
          1: ("Standard", "Check every bill with the rules; field-verify REVIEW lines"),
          2: ("Close watch", "Field-verify every bill; monthly performance meeting; consider holding retainage")}


def package_features(tables, gates, exceptions, pt=None):
    inv = tables["invoices"]
    lines = tables["invoice_lines"]
    pol, co = tables["po_lines"], tables["change_orders"]
    g = gates.merge(inv[["invoice_id", "po_id"]], on="invoice_id")
    contract = pol[pol.line_source == "CONTRACT"].groupby("po_id").scheduled_value.sum()
    cos = co[co.status.isin(["APPROVED", "PENDING"])].groupby("po_id").co_amount.sum()
    tm = lines[lines.line_type == "TM"].merge(inv[["invoice_id", "po_id"]], on="invoice_id").groupby("po_id").amount.sum()
    rows = []
    for po_id, d in g.sort_values("invoice_date").groupby("po_id"):
        billed = d.gross_amount.sum()
        jumps = np.log(d.gross_amount.clip(lower=1)).diff().abs().dropna()
        rows.append({"po_id": po_id, "project_id": d.project_id.iloc[0], "prime_id": d.prime_id.iloc[0], "bills": len(d),
                     "first_pass_pct": 100 * (d.gate == "PASS").mean(), "risk_pct": 100 * d.amount_at_risk.sum() / max(billed, 1),
                     "co_share_pct": 100 * cos.get(po_id, 0) / max(contract.get(po_id, 1), 1),
                     "tm_share_pct": 100 * tm.get(po_id, 0) / max(billed, 1), "avg_jump": float(jumps.mean()) if len(jumps) else 0.0})
    df = pd.DataFrame(rows)
    if pt is not None and "spi" in pt:
        df["spi"] = df.project_id.map(pt.set_index("project_id").spi).fillna(1.0)
    else:
        df["spi"] = 1.0
    return df


def segments(df, k=3, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    feats = list(SEG_FEATURES)
    X = StandardScaler().fit_transform(df[feats].fillna(0))
    sil = {}
    for kk in range(2, min(6, len(df) - 1) + 1):
        lab = KMeans(kk, n_init=20, random_state=seed).fit_predict(X)
        sil[kk] = silhouette_score(X, lab)
    km = KMeans(k, n_init=20, random_state=seed).fit(X)
    df = df.copy()
    df["cluster"] = km.labels_
    # order clusters by risk: low risk -> light touch
    risk = df.groupby("cluster").apply(lambda d: (100 - d.first_pass_pct).mean() + 5 * d.risk_pct.mean() + 20 * (1 - d.spi).mean(),
                                       include_groups=False).sort_values()
    rank = {c: i for i, c in enumerate(risk.index)}
    df["segment_rank"] = df.cluster.map(rank)
    df["segment"] = df.segment_rank.map(lambda r: POLICY[min(r, 2)][0])
    df["policy"] = df.segment_rank.map(lambda r: POLICY[min(r, 2)][1])
    from sklearn.decomposition import PCA
    xy = PCA(2, random_state=seed).fit_transform(X)
    df["pc1"], df["pc2"] = xy[:, 0], xy[:, 1]
    prof = df.groupby("segment")[feats + ["bills"]].mean().reindex([POLICY[i][0] for i in range(3)]).dropna(how="all")
    prof["packages"] = df.groupby("segment").size()
    return df, prof, pd.Series(sil, name="silhouette")


# ---------------------------------------------------------------- F14
def issue_baskets(gates, exceptions, label=None):
    s = exceptions.groupby("invoice_id").rule.apply(lambda r: sorted(set(r)))
    b = gates[["invoice_id"]].merge(s.rename("issues").reset_index(), on="invoice_id", how="left")
    b["issues"] = b.issues.apply(lambda x: x if isinstance(x, list) else [])
    return b


def association_rules(baskets, min_support=0.02, min_conf=0.2):
    n = len(baskets)
    sets = [set(x) for x in baskets.issues]
    items = sorted({i for s in sets for i in s})
    sup1 = {i: sum(i in s for s in sets) / n for i in items}
    rows = []
    for a, b in combinations(items, 2):
        both = sum((a in s) and (b in s) for s in sets) / n
        if both < min_support:
            continue
        for x, y in [(a, b), (b, a)]:
            conf = both / sup1[x]
            lift = conf / sup1[y]
            if conf >= min_conf:
                # Fisher exact test on the 2x2 table
                n11 = int(round(both * n)); n1_ = int(round(sup1[x] * n)); n_1 = int(round(sup1[y] * n))
                tab = [[n11, n1_ - n11], [n_1 - n11, n - n1_ - n_1 + n11]]
                p = stats.fisher_exact(tab, alternative="greater")[1]
                rows.append({"if_bill_has": x, "then_also": y, "support": both, "confidence": conf, "lift": lift,
                             "bills_with_both": n11, "p_value": p})
    singles = pd.DataFrame({"issue": items, "support": [sup1[i] for i in items]})
    return pd.DataFrame(rows).sort_values("lift", ascending=False) if rows else pd.DataFrame(), singles


# ---------------------------------------------------------------- F15
GRANGER_VARS = {"co_approved_m": "Change orders approved ($M)", "error_rate": "Share of bills with an error",
                "overbilled_k": "Overbilled ($K)", "jump": "Billing jump (avg log change)"}


def granger_panel():
    """Monthly panel per (program, contractor) from the 12 training programs: CO approvals, billing jumps, errors."""
    f = pd.read_csv(TRAIN / "bill_features.csv")
    f = f[f.split == "train"]
    p = (f.groupby(["program", "prime_id", "period"])
          .agg(error_rate=("has_error", "mean"), overbilled_k=("overbilled", lambda s: s.sum() / 1e3),
               jump=("jump_vs_prev", "mean"), bills=("invoice_id", "size")).reset_index())
    co_file = TRAIN / "co_monthly.csv"
    if co_file.exists():
        co = pd.read_csv(co_file)
        p = p.merge(co, on=["program", "prime_id", "period"], how="left")
        p["co_approved_m"] = p.co_approved.fillna(0) / 1e6
    else:
        p["co_approved_m"] = 0.0
    return p.sort_values(["program", "prime_id", "period"])


def pooled_baskets():
    f = TRAIN / "bill_issues.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f).fillna({"issues": ""})
    b["issues"] = b.issues.apply(lambda x: [i for i in str(x).split("|") if i])
    return b


def _lags(p, col, L):
    g = p.groupby(["program", "prime_id"])[col]
    return pd.concat({f"{col}_l{l}": g.shift(l) for l in range(1, L + 1)}, axis=1)


def granger(p, cause, effect, L=1, time_fe=True):
    """F-test: do lags of `cause` improve the prediction of `effect` beyond lags of `effect`? Pooled panel OLS.
    time_fe adds month dummies so a shared program life-cycle (both series rising mid-project) is not mistaken for a lead."""
    from cost_intel import ols
    d = pd.concat([p[[effect, "period"]], _lags(p, effect, L), _lags(p, cause, L)], axis=1).dropna()
    y = d[effect].values
    fe = pd.get_dummies(d.period, drop_first=True).astype(float).values if time_fe else np.empty((len(d), 0))
    Xr = np.column_stack([np.ones(len(d)), fe, d[[f"{effect}_l{l}" for l in range(1, L + 1)]].values])
    Xu = np.column_stack([Xr, d[[f"{cause}_l{l}" for l in range(1, L + 1)]].values])
    r, u = ols(y, Xr, [f"r{i}" for i in range(Xr.shape[1])]), ols(y, Xu, [f"u{i}" for i in range(Xu.shape[1])])
    F = ((r["sse"] - u["sse"]) / L) / (u["sse"] / (u["n"] - u["k"]))
    pval = float(stats.f.sf(F, L, u["n"] - u["k"]))
    coef = u["coef"].iloc[-L:].estimate.sum()
    return {"cause": cause, "effect": effect, "lags": L, "F": float(F), "p_value": pval, "n": u["n"], "sum_of_cause_coefs": float(coef),
            "time_fe": time_fe}
