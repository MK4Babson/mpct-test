#!/usr/bin/env bash
# Load the control-tower tables into BigQuery and run the rules engine.
# Needs: gcloud SDK (bq CLI), a GCP project with BigQuery enabled, and the CSVs in DATA_DIR.
# Usage: PROJECT=my-gcp-project DATASET=control_tower DATA_DIR=../data/raw ./load_and_run.sh
set -euo pipefail
PROJECT=${PROJECT:?set PROJECT}
DATASET=${DATASET:-control_tower}
DATA_DIR=${DATA_DIR:-../data/raw}
LOCATION=${LOCATION:-US}
bq --location="$LOCATION" mk -f --dataset "$PROJECT:$DATASET" >/dev/null
for t in purchase_orders po_lines change_orders tm_preapprovals goods_receipts invoices invoice_lines primes projects; do
  if [ -f "$DATA_DIR/$t.csv" ]; then
    bq load --replace --skip_leading_rows=1 --source_format=CSV "$PROJECT:$DATASET.$t" "$DATA_DIR/$t.csv" "schemas/$t.json"
  fi
done
# primes and projects (optional reference tables, schema auto-detected)
for t in primes projects; do
  [ -f "$DATA_DIR/$t.csv" ] && bq load --replace --autodetect --source_format=CSV "$PROJECT:$DATASET.$t" "$DATA_DIR/$t.csv"
done
# tolerances (edit to change policy)
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" "CREATE OR REPLACE TABLE params AS SELECT 1.5 AS qty_review_pct, 5.0 AS qty_hold_pct, 0.5 AS rate_hold_pct, 1.0 AS pct_review_pts, 5.0 AS pct_hold_pts, 0.0 AS tm_review_pct, 10.0 AS tm_hold_pct, 1.0 AS money_tol, 178.0 AS per_diem_cap"
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" < engine_bigquery.sql
bq query --use_legacy_sql=false --dataset_id="$PROJECT:$DATASET" < reporting_views.sql
echo "Done. Tables: $PROJECT:$DATASET.exceptions, invoice_gates; views: v_gates, v_exceptions, v_contractor_month"
