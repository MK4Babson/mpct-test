# Looker Studio report on BigQuery

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
