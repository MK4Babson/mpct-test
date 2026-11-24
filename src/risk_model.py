"""
Bill risk model: predicts the probability that a bill contains a billing error *before* anyone checks it,
so reviewers can start with the riskiest bills.

Features use only information available before the check (no leakage from the rules engine's verdict
on the same bill): contractor and bill history, size and progress patterns, change-order activity.
The previous bills' rule outcomes are allowed, because they are already known when a new bill arrives.

Models (as in the ML course): logistic regression, CART, random forest, gradient boosting, and a stack.
Evaluation: hold-out program never seen in training; AUC, ROC, lift / gains, confusion matrix.
Reviewer plan: 0-1 knapsack (MILP, SciPy HiGHS) maximizing expected overbilling caught within reviewer hours.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "training" / "bill_features.csv"

FEATURES = [
    "bill_seq", "early_bill", "cum_billed_pct_prev", "bill_pct_of_contract", "jump_vs_prev",
    "n_lines", "share_ls_lines", "n_tm_lines", "n_co_lines", "has_per_diem",
    "pending_cos_on_po", "prior_issue_rate", "prior_bill_flagged", "closeout",
    "prime_electrical", "prime_mechanical", "prime_civil",
]
LABELS = {
    "bill_seq": "later bill in the contract", "early_bill": "one of the first two bills",
    "cum_billed_pct_prev": "share of contract already billed", "bill_pct_of_contract": "large bill for this contract",
    "jump_vs_prev": "big jump vs the previous bill", "n_lines": "many lines", "share_ls_lines": "many lump-sum lines",
    "n_tm_lines": "time & materials lines", "n_co_lines": "change-order lines", "has_per_diem": "per diem billed",
    "pending_cos_on_po": "pending change orders on the contract", "prior_issue_rate": "contractor's past issue rate",
    "prior_bill_flagged": "previous bill was flagged", "closeout": "close-out stage (over 75% billed)",
    "prime_electrical": "electrical package", "prime_mechanical": "mechanical package", "prime_civil": "civil package",
}


# ---------------------------------------------------------------- features
def bill_features(tables, gates, truth=None):
    """One row per bill (duplicates included), features known before checking it, plus label if truth given."""
    inv = tables["invoices"].copy()
    lines = tables["invoice_lines"]
    po = tables["purchase_orders"]
    pol = tables["po_lines"]
    contract = pol[pol.line_source == "CONTRACT"].groupby("po_id").scheduled_value.sum().rename("contract_value")
    co = tables["change_orders"]
    pend = co[co.status == "PENDING"].groupby("po_id").size().rename("pending_cos_on_po")
    lg = lines.groupby("invoice_id").agg(
        n_lines=("line_no", "size"),
        n_ls=("unit", lambda s: int((s.astype(str).str.upper() == "LS").sum())),
        n_tm_lines=("line_type", lambda s: int((s == "TM").sum())),
        n_co_lines=("line_type", lambda s: int((s == "CO").sum())),
        has_per_diem=("line_type", lambda s: int((s == "PER_DIEM").any())))
    work = lines[lines.line_type.isin(["SOV", "CO"])].groupby("invoice_id").amount.sum().rename("work_amount")
    df = (inv.merge(lg, left_on="invoice_id", right_index=True, how="left")
             .merge(work, left_on="invoice_id", right_index=True, how="left")
             .merge(contract, left_on="po_id", right_index=True, how="left")
             .merge(pend, left_on="po_id", right_index=True, how="left"))
    df = df.merge(gates[["invoice_id", "gate"]], on="invoice_id", how="left")
    df["work_amount"] = df.work_amount.fillna(0)
    df["pending_cos_on_po"] = df.pending_cos_on_po.fillna(0)
    df["share_ls_lines"] = df.n_ls / df.n_lines.clip(lower=1)
    df = df.sort_values(["po_id", "invoice_date", "invoice_id"])
    g = df.groupby("po_id")
    df["bill_seq"] = g.cumcount() + 1
    df["early_bill"] = (df.bill_seq <= 2).astype(int)
    df["cum_billed_pct_prev"] = (g.work_amount.cumsum() - df.work_amount) / df.contract_value.replace(0, np.nan)
    df["bill_pct_of_contract"] = df.work_amount / df.contract_value.replace(0, np.nan)
    prev = g.gross_amount.shift(1)
    df["jump_vs_prev"] = np.log((df.gross_amount.clip(lower=1)) / prev.clip(lower=1)).fillna(0)
    flagged = (df.gate != "PASS").astype(float)
    prior_flags = flagged.groupby(df.po_id).cumsum() - flagged
    df["prior_issue_rate"] = prior_flags / (df.bill_seq - 1).replace(0, np.nan)
    df["prior_issue_rate"] = df.prior_issue_rate.fillna(0.3)  # no history yet: assume the average rate
    df["prior_bill_flagged"] = flagged.groupby(df.po_id).shift(1).fillna(0)
    df["closeout"] = (df.cum_billed_pct_prev > 0.75).astype(int)
    kind = df.prime_id.astype(str).str[-1]
    df["prime_electrical"] = (kind == "E").astype(int)
    df["prime_mechanical"] = (kind == "M").astype(int)
    df["prime_civil"] = (kind == "C").astype(int)
    df[FEATURES] = df[FEATURES].fillna(0).astype(float)
    if truth is not None:
        t = truth.copy()
        t["over"] = pd.to_numeric(t.overbilled_amount, errors="coerce").fillna(0)
        lab = t.groupby("invoice_id").agg(n_errors=("error_type", "size"), overbilled=("over", "sum"))
        df = df.merge(lab, left_on="invoice_id", right_index=True, how="left")
        df["has_error"] = df.n_errors.notna().astype(int)
        df["overbilled"] = df.overbilled.fillna(0)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- models
def make_models():
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cart = DecisionTreeClassifier(max_depth=4, min_samples_leaf=25, random_state=0)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0)
    stack = StackingClassifier(estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                                           ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
                                           ("gbm", GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0))],
                               final_estimator=LogisticRegression(max_iter=2000), cv=5)
    return {"Logistic regression": lr, "CART (depth 4)": cart, "Random forest": rf, "Gradient boosting": gbm, "Stacked ensemble": stack}


def train_and_evaluate(train, holdout):
    """Fit every model on the training programs; evaluate on the hold-out program."""
    from sklearn.metrics import roc_auc_score, roc_curve
    X, y = train[FEATURES].values, train.has_error.values
    Xh, yh = holdout[FEATURES].values, holdout.has_error.values
    models, results, rocs, scores = make_models(), [], [], {}
    for name, m in models.items():
        m.fit(X, y)
        p = m.predict_proba(Xh)[:, 1]
        scores[name] = p
        fpr, tpr, _ = roc_curve(yh, p)
        rocs.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
        g = gains(holdout.assign(score=p))
        top20 = g.loc[(g.share_reviewed - 0.2).abs().idxmin()]
        results.append({"model": name, "auc": roc_auc_score(yh, p), "errors_caught_top20": top20.share_errors,
                        "money_caught_top20": top20.share_money})
    res = pd.DataFrame(results).sort_values("auc", ascending=False)
    return models, res, pd.concat(rocs), scores


def gains(df, score="score"):
    """Cumulative share of error bills and overbilled money caught as bills are reviewed in score order."""
    d = df.sort_values(score, ascending=False).reset_index(drop=True)
    n, e, mny = len(d), d.has_error.sum(), d.overbilled.sum()
    out = pd.DataFrame({"share_reviewed": (np.arange(n) + 1) / n,
                        "share_errors": d.has_error.cumsum() / max(e, 1),
                        "share_money": d.overbilled.cumsum() / max(mny, 1)})
    return pd.concat([pd.DataFrame({"share_reviewed": [0.0], "share_errors": [0.0], "share_money": [0.0]}), out])


def confusion(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return pd.DataFrame({"Predicted risky": [int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())],
                         "Predicted low risk": [int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())]},
                        index=["Has error", "Clean"])


def reasons(lr_pipeline, X_row_df, k=3):
    """Top drivers for one bill from the logistic regression (coefficient x standardized value)."""
    sc, lr = lr_pipeline.steps[0][1], lr_pipeline.steps[-1][1]
    z = (X_row_df[FEATURES].values - sc.mean_) / np.where(sc.scale_ == 0, 1, sc.scale_)
    contrib = z * lr.coef_[0]
    out = []
    for row in contrib:
        idx = np.argsort(-row)[:k]
        out.append("; ".join(LABELS[FEATURES[i]] for i in idx if row[i] > 0.05) or "no strong risk signals")
    return out


def lr_coefficients(lr_pipeline):
    lr = lr_pipeline.steps[-1][1]
    return pd.DataFrame({"feature": [LABELS[f] for f in FEATURES], "coef": lr.coef_[0],
                         "odds_ratio_per_sd": np.exp(lr.coef_[0])}).sort_values("coef", ascending=False)


def tree_rules(cart):
    from sklearn.tree import export_text
    return export_text(cart, feature_names=[LABELS[f] for f in FEATURES], decimals=2)


# ---------------------------------------------------------------- reviewer plan
def review_plan(df, hours, base_hours=0.5, per_line=0.08):
    """Choose bills to review within `hours`, maximizing expected overbilling caught (0-1 knapsack, MILP)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    d = df.copy()
    d["review_hours"] = base_hours + per_line * d.n_lines
    value = d.expected_money.values
    if len(d) == 0 or value.sum() <= 0:
        d["review"] = False
        return d
    res = milp(c=-value, constraints=LinearConstraint(d.review_hours.values[None, :], ub=hours),
               integrality=np.ones(len(d)), bounds=Bounds(0, 1), options={"time_limit": 10})
    pick = np.round(res.x).astype(bool) if res.x is not None else np.zeros(len(d), bool)
    d["review"] = pick
    return d
