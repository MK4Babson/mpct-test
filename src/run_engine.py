"""
Run the 3-way match + governance gate engine (sql/engine.sql) over a data folder.

Usage:
  python src/run_engine.py                                  # demo data in data/raw
  python src/run_engine.py --data data/my_project --out outputs/my_project

Writes: exceptions.csv, invoice_gates.csv, prime_scorecard.csv to the --out folder.
Note: the engine never reads ground_truth_errors.csv. Only evaluate.py uses that file.
"""
import argparse
from pathlib import Path
import duckdb
import pandas as pd

from config import TOLERANCES, PER_DIEM_CAP
from schema import SCHEMA, SQL_TYPES, coerce, read_folder

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, SQL = ROOT / "data" / "raw", ROOT / "outputs", ROOT / "sql" / "engine.sql"


def _load(con, name, df):
    cols = SCHEMA[name]
    ddl = ", ".join(f"{c} {SQL_TYPES[t]}" for c, t, *_ in cols)
    con.execute(f"CREATE OR REPLACE TABLE {name} ({ddl})")
    if df is None or len(df) == 0:
        return
    df = coerce(df, name)[[c for c, *_ in cols]]
    con.register("_tmp", df)
    sel = ", ".join(f"CAST({c} AS {SQL_TYPES[t]})" for c, t, *_ in cols)
    con.execute(f"INSERT INTO {name} SELECT {sel} FROM _tmp")
    con.unregister("_tmp")


def run(tolerances=None, data_dir=RAW, tables=None, per_diem_cap=PER_DIEM_CAP):
    """Run the engine. Pass either a data folder or a dict of DataFrames (tables)."""
    tol = {**TOLERANCES, **(tolerances or {})}
    tables = tables if tables is not None else read_folder(data_dir)
    con = duckdb.connect()
    for name in SCHEMA:
        _load(con, name, tables.get(name))
    con.register("params_df", pd.DataFrame([{**tol, "per_diem_cap": per_diem_cap}]))
    con.execute("CREATE OR REPLACE TABLE params AS SELECT * FROM params_df")
    con.execute(SQL.read_text())
    exceptions = con.execute("SELECT * FROM exceptions ORDER BY invoice_id, line_no").df()
    gates = con.execute("SELECT * FROM invoice_gates ORDER BY invoice_id").df()
    con.close()
    # A duplicate invoice's risk is the whole invoice; don't double count its line issues
    gates["amount_at_risk"] = gates[["amount_at_risk", "gross_amount"]].min(axis=1)
    return exceptions, gates


def scorecard(gates, exceptions, primes=None):
    g = gates.groupby("prime_id").agg(
        invoices=("invoice_id", "count"),
        billed_gross=("gross_amount", "sum"),
        pass_n=("gate", lambda s: (s == "PASS").sum()),
        review_n=("gate", lambda s: (s == "REVIEW").sum()),
        hold_n=("gate", lambda s: (s == "HOLD").sum()),
        amount_at_risk=("amount_at_risk", "sum"),
    ).reset_index()
    g["first_pass_rate_pct"] = (100 * g["pass_n"] / g["invoices"]).round(1)
    g["risk_pct_of_billed"] = (100 * g["amount_at_risk"] / g["billed_gross"].where(g["billed_gross"] != 0)).round(2)
    if len(exceptions):
        top = (exceptions.merge(gates[["invoice_id", "prime_id"]], on="invoice_id")
               .groupby(["prime_id", "rule"]).size().reset_index(name="n")
               .sort_values(["prime_id", "n"], ascending=[True, False])
               .groupby("prime_id").head(1).rename(columns={"rule": "top_exception"}))
        g = g.merge(top[["prime_id", "top_exception"]], on="prime_id", how="left")
    else:
        g["top_exception"] = None
    if primes is not None and len(primes):
        g = primes[["prime_id", "prime_name"]].merge(g, on="prime_id", how="right")
    return g


def main():
    ap = argparse.ArgumentParser(description="Run the governance engine on a data folder")
    ap.add_argument("--data", default=str(RAW), help="folder with the input CSVs")
    ap.add_argument("--out", default=str(OUT), help="folder for results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables = read_folder(args.data)
    exceptions, gates = run(tables=tables)
    sc = scorecard(gates, exceptions, tables.get("primes"))
    exceptions.to_csv(out / "exceptions.csv", index=False)
    gates.to_csv(out / "invoice_gates.csv", index=False)
    sc.to_csv(out / "prime_scorecard.csv", index=False)

    print("Governance gate results")
    print(gates["gate"].value_counts().to_string())
    print(f"\nTotal billed (gross): ${gates['gross_amount'].sum():,.0f}")
    print(f"Amount at risk held/flagged: ${gates['amount_at_risk'].sum():,.0f}")
    if len(exceptions):
        print("\nExceptions by rule:")
        print(exceptions.groupby(["rule", "severity"]).size().unstack(fill_value=0).to_string())
    print("\nPrime scorecard:")
    print(sc[["prime_id", "invoices", "pass_n", "review_n", "hold_n", "first_pass_rate_pct",
              "amount_at_risk", "risk_pct_of_billed", "top_exception"]].to_string(index=False))
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
