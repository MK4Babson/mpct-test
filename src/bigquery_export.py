"""
F23: package the rules engine for Google BigQuery + Looker Studio.

Writes bigquery/:
  engine_bigquery.sql   the same rules, translated from DuckDB to BigQuery Standard SQL
  reporting_views.sql   views Looker Studio reads (gates by project/month, exceptions with labels, contractor scorecard)
  schemas/*.json        BigQuery table schemas, one per input table
  load_and_run.sh       creates the dataset, loads the CSVs, runs the engine and the views (bq CLI)
  LOOKER_STUDIO.md      how to build the report on top of the views

Usage:  python src/bigquery_export.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import SCHEMA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bigquery"
BQ_TYPE = {"str": "STRING", "float": "FLOAT64", "int": "INT64", "date": "DATE"}


def translate(sql):
    s = sql
    s = s.replace("AS VARCHAR)", "AS STRING)")
    s = re.sub(r",\s*'g'\)", ")", s)  # BigQuery REGEXP_REPLACE is global and takes no flags
    s = re.sub(r"DATE_DIFF\('day',\s*([\w\.]+),\s*([\w\.]+)\)", r"DATE_DIFF(\2, \1, DAY)", s)
    s = s.replace("-- Standard SQL (runs in DuckDB locally; written to port to BigQuery).",
                  "-- BigQuery Standard SQL. Generated from sql/engine.sql by src/bigquery_export.py; run as one script.")
    return s


REPORTING = """-- Reporting views for Looker Studio (BigQuery Standard SQL)
CREATE OR REPLACE VIEW v_gates AS
SELECT g.*, p.prime_name, pr.short_name AS project_name,
       CASE g.gate WHEN 'PASS' THEN 1 ELSE 0 END AS is_pass,
       CASE g.gate WHEN 'HOLD' THEN g.amount_at_risk ELSE 0 END AS blocked,
       CASE g.gate WHEN 'REVIEW' THEN g.amount_at_risk ELSE 0 END AS to_check
FROM invoice_gates g
LEFT JOIN primes p ON p.prime_id = g.prime_id
LEFT JOIN projects pr ON pr.project_id = g.project_id;

CREATE OR REPLACE VIEW v_exceptions AS
SELECT e.*, g.project_id, g.prime_id, g.period,
       CASE e.rule
         WHEN 'QTY_OVER_RECEIPT' THEN 'Quantity above field-verified'
         WHEN 'RATE_OVER_CONTRACT' THEN 'Unit rate above contract'
         WHEN 'PCT_COMPLETE_OVERBILL' THEN 'Lump sum ahead of verified progress'
         WHEN 'UNAPPROVED_CO' THEN 'Unapproved change order'
         WHEN 'TM_OVER_APPROVED' THEN 'T&M hours above approved'
         WHEN 'TM_NOT_PREAPPROVED' THEN 'T&M with no pre-approval'
         WHEN 'PER_DIEM_OVER_POLICY' THEN 'Per diem above policy'
         WHEN 'RETAINAGE_ERROR' THEN 'Retainage error'
         WHEN 'EXTENSION_ERROR' THEN 'Arithmetic error'
         WHEN 'DUPLICATE_INVOICE' THEN 'Duplicate invoice'
         WHEN 'NO_RECEIPT' THEN 'No field verification'
         WHEN 'HEADER_MISMATCH' THEN 'Totals do not add up'
         ELSE e.rule END AS issue
FROM exceptions e JOIN invoice_gates g ON g.invoice_id = e.invoice_id;

CREATE OR REPLACE VIEW v_contractor_month AS
SELECT prime_id, period, COUNT(*) AS bills, AVG(CASE gate WHEN 'PASS' THEN 1.0 ELSE 0 END) AS first_pass_rate,
       SUM(gross_amount) AS billed, SUM(amount_at_risk) AS money_at_risk
FROM invoice_gates GROUP BY prime_id, period;
"""

LOAD = """#!/usr/bin/env bash
# Load the control-tower tables into BigQuery and run the rules engine.
# Needs: gcloud SDK (bq CLI), a GCP project with BigQuery enabled, and the CSVs in DATA_DIR.
# Usage: PROJECT=my-gcp-project DATASET=control_tower DATA_DIR=../data/raw ./load_and_run.sh
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT}
DATASET=${DATASET:-control_tower}
DATA_DIR=${DATA_DIR:-../data/raw}
LOCATION=${LOCATION:-US}
bq --location="$LOCATION" mk -f --dataset "$PROJECT:$DATASET" >/dev/null
for t in TABLES; do
  if [ -f "$DATA_DIR/$t.csv" ]; then
    bq load --replace --skip_leading_rows=1 --source_format=CSV "$PROJECT:$DATASET.$t" "$DATA_DIR/$t.csv" "schemas/$t.json"
  fi
done
# primes and projects (optional reference tables, schema auto-detected)
for t in primes projects; do
  [ -f "$DATA_DIR/$t.csv" ] && bq load --replace --autodetect --source_format=CSV "$PROJECT:$DATASET.$t" "$DATA_DIR/$t.csv"
done
# tolerances (edit to change policy)
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" "CREATE OR REPLACE TABLE params AS SELECT PARAMS"
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" < engine_bigquery.sql
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" < reporting_views.sql
echo "Done. Tables: $PROJECT:$DATASET.exceptions, invoice_gates; views: v_gates, v_exceptions, v_contractor_month"
"""

LOOKER = """# Looker Studio report on BigQuery

1. Run `load_and_run.sh` (see the header for variables). It loads the CSVs, runs the same rules the app uses and creates three views.
2. Open https://lookerstudio.google.com → **Create → Report → Add data → BigQuery** → your project → dataset `control_tower`.
   Add `v_gates`, `v_exceptions` and `v_contractor_month` as data sources.
3. Build four pages (mirrors the app):

| Page | Chart | Data source | Dimension | Metric |
|---|---|---|---|---|
| Overview | Scorecards | v_gates | – | SUM(gross_amount), SUM(blocked), SUM(to_check), AVG(is_pass) |
| Overview | Bar | v_gates | project_name | SUM(blocked), SUM(to_check) |
| Issues | Bar (sorted) | v_exceptions | issue | SUM(amount_at_risk), COUNT(invoice_id) |
| Contractors | Time series | v_contractor_month | period, prime_id | AVG(first_pass_rate) |
| Bills | Table + filters | v_gates | invoice_number, project_name, prime_name, period, gate | gross_amount, amount_at_risk |

4. Add controls: drop-down for `project_name`, `prime_name`, date range on `invoice_date`. Share with **Viewer** access.
5. Schedule: re-run step 1 when new bills arrive (Cloud Scheduler + a Cloud Run job calling the script), or use BigQuery
   scheduled queries for the engine once the input tables are fed by Sheets/Drive connectors.

The engine SQL is translated automatically from `sql/engine.sql` (CAST AS STRING, BigQuery REGEXP_REPLACE, DATE_DIFF argument order).
Syntax is validated with sqlglot's BigQuery parser; run it on your own project before relying on it.
"""


def main():
    from config import PER_DIEM_CAP, TOLERANCES
    OUT.mkdir(exist_ok=True)
    (OUT / "schemas").mkdir(exist_ok=True)
    eng = translate((ROOT / "sql" / "engine.sql").read_text())
    (OUT / "engine_bigquery.sql").write_text(eng)
    (OUT / "reporting_views.sql").write_text(REPORTING)
    for t, cols in SCHEMA.items():
        (OUT / "schemas" / f"{t}.json").write_text(json.dumps(
            [{"name": c, "type": BQ_TYPE.get(typ, "STRING"), "mode": "NULLABLE"} for c, typ, *_ in cols], indent=1))
    params = ", ".join(f"{float(v)} AS {k}" for k, v in {**TOLERANCES, "per_diem_cap": PER_DIEM_CAP}.items())
    (OUT / "load_and_run.sh").write_text(LOAD.replace("TABLES", " ".join(SCHEMA)).replace("PARAMS", params))
    (OUT / "load_and_run.sh").chmod(0o755)
    (OUT / "LOOKER_STUDIO.md").write_text(LOOKER)
    try:
        import sqlglot
        n = 0
        for f in ["engine_bigquery.sql", "reporting_views.sql"]:
            for stmt in sqlglot.parse((OUT / f).read_text(), read="bigquery"):
                n += stmt is not None
        print(f"sqlglot: parsed {n} BigQuery statements without errors")
    except ImportError:
        print("sqlglot not installed: skipped syntax validation")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
