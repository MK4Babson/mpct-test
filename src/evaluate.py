"""
Score the engine against the planted ground truth.

  * Precision = of the issues the engine flagged, how many were real errors
  * Recall    = of the real errors, how many the engine caught
  * Tolerance sweep = how the quantity tolerance trades missed errors against false alarms

Usage:  python src/evaluate.py
Writes: outputs/evaluation_by_rule.csv, outputs/evaluation_report.md, outputs/tolerance_tradeoff.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_engine import run
from config import TOLERANCES

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "outputs"

# Palette: validated default categorical slots 1-2, light surface
BLUE, ORANGE, SURFACE = "#2a78d6", "#eb6834", "#fcfcfb"
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"


def keys(df, type_col):
    ln = df["line_no"].fillna(-1).astype(int)
    return set(zip(df["invoice_id"], ln, df[type_col]))


def score(exceptions, truth):
    det = keys(exceptions, "rule")
    tru = keys(truth, "error_type")
    rules = sorted(set(truth["error_type"]) | set(exceptions["rule"]))
    rows = []
    for r in rules:
        d = {k for k in det if k[2] == r}
        t = {k for k in tru if k[2] == r}
        tp, fp, fn = len(d & t), len(d - t), len(t - d)
        rows.append({"rule": r, "planted": len(t), "flagged": len(d), "true_pos": tp,
                     "false_pos": fp, "missed": fn,
                     "precision_pct": round(100 * tp / (tp + fp), 1) if tp + fp else np.nan,
                     "recall_pct": round(100 * tp / (tp + fn), 1) if tp + fn else np.nan})
    tp, fp, fn = len(det & tru), len(det - tru), len(tru - det)
    rows.append({"rule": "ALL RULES", "planted": len(tru), "flagged": len(det), "true_pos": tp,
                 "false_pos": fp, "missed": fn,
                 "precision_pct": round(100 * tp / (tp + fp), 1),
                 "recall_pct": round(100 * tp / (tp + fn), 1)})
    return pd.DataFrame(rows), det - tru, tru - det


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Score the engine against ground_truth_errors.csv")
    ap.add_argument("--data", default=str(RAW))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(data / "ground_truth_errors.csv", dtype={"invoice_id": str, "error_type": str})
    exceptions, gates = run(data_dir=data)
    by_rule, fps, fns = score(exceptions, truth)
    by_rule.to_csv(out / "evaluation_by_rule.csv", index=False)

    # Invoice-level confusion: does the gate stop invoices that contain any error?
    bad = set(truth["invoice_id"])
    gates["has_error"] = gates["invoice_id"].isin(bad)
    gates["flagged"] = gates["gate"] != "PASS"
    cm = pd.crosstab(gates["has_error"].map({True: "Has planted error", False: "Clean"}),
                     gates["gate"]).reindex(columns=["PASS", "REVIEW", "HOLD"], fill_value=0)

    # Dollars caught vs missed
    t = truth.copy()
    t["key"] = list(zip(t["invoice_id"], t["line_no"].fillna(-1).astype(int), t["error_type"]))
    t["caught"] = ~t["key"].isin(fns)
    dollars_caught = t.loc[t.caught, "overbilled_amount"].sum()
    dollars_total = t["overbilled_amount"].sum()

    # Missed / false alarm details (for honest reporting)
    fn_df = t[~t.caught][["invoice_id", "line_no", "error_type", "overbilled_amount"]]
    fp_df = exceptions[[k in fps for k in keys_iter(exceptions)]]

    # ---- Tolerance sweep for the quantity rule ----
    sweep = []
    q_truth = truth[truth.error_type == "QTY_OVER_RECEIPT"]
    for tol in np.arange(0.0, 6.01, 0.25):
        ex, _ = run(data_dir=data, tolerances={"qty_review_pct": float(tol), "qty_hold_pct": max(float(tol), TOLERANCES["qty_hold_pct"])})
        r, _, _ = score(ex[ex.rule == "QTY_OVER_RECEIPT"], q_truth)
        r = r[r.rule == "QTY_OVER_RECEIPT"].iloc[0]
        sweep.append({"tolerance_pct": tol, "precision_pct": r.precision_pct,
                      "recall_pct": r.recall_pct, "false_pos": r.false_pos, "missed": r.missed})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "tolerance_sweep.csv", index=False)
    plot_sweep(sweep, out)

    # ---- Report ----
    lines = [
        "# Evaluation report (synthetic data)",
        "",
        f"Engine run with default tolerances: `{TOLERANCES}`",
        "",
        "## Detection by rule",
        "",
        by_rule.to_markdown(index=False),
        "",
        "## Invoice-level gate outcome",
        "",
        cm.to_markdown(),
        "",
        f"**Planted overbilling caught:** ${dollars_caught:,.0f} of ${dollars_total:,.0f} "
        f"({100 * dollars_caught / dollars_total:.1f}%)",
        "",
        "## Missed errors",
        "",
        fn_df.to_markdown(index=False) if len(fn_df) else "None.",
        "",
        "## False alarms",
        "",
        fp_df[["invoice_id", "line_no", "rule", "severity", "variance", "amount_at_risk", "detail"]].to_markdown(index=False)
        if len(fp_df) else "None.",
        "",
        "## Quantity tolerance trade-off",
        "",
        sweep.to_markdown(index=False),
        "",
        "![tolerance trade-off](tolerance_tradeoff.png)",
    ]
    (out / "evaluation_report.md").write_text("\n".join(lines))

    print(by_rule.to_string(index=False))
    print("\nInvoice-level:\n" + cm.to_string())
    print(f"\nOverbilling caught: ${dollars_caught:,.0f} / ${dollars_total:,.0f}")
    print("\nMissed:\n" + (fn_df.to_string(index=False) if len(fn_df) else "none"))
    print("\nFalse alarms:\n" + (fp_df[["invoice_id", "line_no", "rule", "variance", "detail"]].to_string(index=False) if len(fp_df) else "none"))
    print("\nSweep:\n" + sweep.to_string(index=False))


def keys_iter(df):
    return zip(df["invoice_id"], df["line_no"].fillna(-1).astype(int), df["rule"])


def plot_sweep(sweep, out):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = sweep["tolerance_pct"]
    ax.plot(x, sweep["precision_pct"], color=BLUE, lw=2, marker="o", ms=4, label="Precision (flags that were real)")
    ax.plot(x, sweep["recall_pct"], color=ORANGE, lw=2, marker="o", ms=4, label="Recall (real errors caught)")
    ax.axvline(TOLERANCES["qty_review_pct"], color=INK_MUTED, lw=1, ls="--")
    ax.text(TOLERANCES["qty_review_pct"] + 0.08, 8, f"Chosen tolerance {TOLERANCES['qty_review_pct']:.1f}%",
            color=INK_MUTED, fontsize=8)
    for col, colr, name in [("precision_pct", BLUE, "Precision"), ("recall_pct", ORANGE, "Recall")]:
        ax.text(x.iloc[-1] + 0.1, sweep[col].iloc[-1], name, color=INK, fontsize=9, va="center")
    ax.set_xlim(0, 6.9)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Quantity tolerance: billed above field-verified (%)", color=INK, fontsize=9)
    ax.set_ylabel("%", color=INK, fontsize=9)
    ax.set_title("Quantity check: tighter tolerance catches more, but raises false alarms",
                 color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    fig.text(0.01, 0.01, "Synthetic data. Field measurements carry +/-1% noise, so honest invoices can exceed verified quantity slightly.",
             color=INK_MUTED, fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "tolerance_tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
