"""
Build the risk model's training set: 12 synthetic programs for training (seeds 101-112) and one hold-out
program (seed 2026) that the models never see. Writes data/training/bill_features.csv.

Usage:  python src/build_training.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from risk_model import FEATURES, TRAIN_FILE, bill_features  # noqa: E402
from run_engine import run  # noqa: E402
from schema import coerce, read_folder  # noqa: E402

TRAIN_SEEDS, HOLDOUT_SEED = list(range(101, 113)), 2026


def main():
    rows, issues, cos = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        for seed in TRAIN_SEEDS + [HOLDOUT_SEED]:
            out = Path(tmp) / f"s{seed}"
            env = {**os.environ, "MPCT_SEED": str(seed)}
            subprocess.run([sys.executable, str(Path(__file__).parent / "generate_data.py"), "--out", str(out)],
                           env=env, check=True, capture_output=True)
            raw = read_folder(out)
            tables = {k: coerce(v, k) for k, v in raw.items()}
            exc, gates = run(tables=tables)
            iss = exc.groupby("invoice_id").rule.apply(lambda r: "|".join(sorted(set(r))))
            b = gates[["invoice_id", "prime_id", "period"]].assign(issues=gates.invoice_id.map(iss).fillna(""), program=f"seed {seed}")
            issues.append(b)
            co = tables["change_orders"].merge(tables["purchase_orders"][["po_id", "prime_id"]], on="po_id", suffixes=("", "_po"))
            co = co[co.status == "APPROVED"].groupby(["prime_id", "approval_period"]).co_amount.sum().reset_index()
            cos.append(co.rename(columns={"approval_period": "period", "co_amount": "co_approved"}).assign(program=f"seed {seed}"))
            truth = pd.read_csv(out / "ground_truth_errors.csv", dtype={"invoice_id": str})
            f = bill_features(tables, gates, truth)
            f["program"] = f"seed {seed}"
            f["split"] = "holdout" if seed == HOLDOUT_SEED else "train"
            rows.append(f)
    df = pd.concat(rows, ignore_index=True)
    keep = ["program", "split", "invoice_id", "prime_id", "period", "gross_amount"] + FEATURES + ["has_error", "overbilled"]
    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(TRAIN_FILE, index=False)
    pd.concat(issues).to_csv(TRAIN_FILE.parent / "bill_issues.csv", index=False)
    pd.concat(cos).to_csv(TRAIN_FILE.parent / "co_monthly.csv", index=False)
    # model selection by 5-fold cross-validation on the training programs only (the hold-out stays untouched)
    from sklearn.model_selection import GroupKFold, cross_val_score
    from risk_model import make_models
    tr = df[df.split == "train"]
    rep = []
    for name, m in make_models().items():
        cv = cross_val_score(m, tr[FEATURES].values, tr.has_error.values, groups=tr.program.values,
                             cv=GroupKFold(n_splits=5), scoring="roc_auc")
        rep.append({"model": name, "cv_auc": cv.mean(), "cv_auc_sd": cv.std()})
    pd.DataFrame(rep).to_csv(TRAIN_FILE.parent / "model_selection.csv", index=False)
    print(pd.DataFrame(rep).round(3).to_string())
    print(f"wrote {len(df)} bills ({(df.split == 'train').sum()} train, {(df.split == 'holdout').sum()} hold-out), "
          f"error rate {df.has_error.mean():.0%} -> {TRAIN_FILE}")


if __name__ == "__main__":
    main()
