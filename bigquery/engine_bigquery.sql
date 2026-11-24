-- =====================================================================
-- Multi-Prime Commercial Control Tower: 3-way match + governance gates
-- BigQuery Standard SQL. Generated from sql/engine.sql by src/bigquery_export.py; run as one script.
-- Inputs : invoices, invoice_lines, po_lines, goods_receipts,
--          change_orders, tm_preapprovals, purchase_orders, params
-- Outputs: exceptions (one row per rule hit), invoice_gates (one row per invoice)
-- =====================================================================

CREATE OR REPLACE VIEW lines_enriched AS
SELECT il.*, i.po_id, i.prime_id, i.project_id, i.period, i.invoice_date
FROM invoice_lines il
JOIN invoices i ON i.invoice_id = il.invoice_id;

-- ---------------------------------------------------------------------
-- RULE 1  RATE_OVER_CONTRACT  (PO leg): billed unit rate vs contract/CO rate
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_rate AS
SELECT l.invoice_id, l.line_no, 'RATE_OVER_CONTRACT' AS rule,
       'HOLD' AS severity,
       ROUND(100.0 * (l.unit_rate - pl.contract_unit_rate) / pl.contract_unit_rate, 2) AS variance,
       ROUND((l.unit_rate - pl.contract_unit_rate) * l.qty, 2) AS amount_at_risk,
       'Billed rate ' || CAST(l.unit_rate AS STRING) || ' vs contract ' || CAST(pl.contract_unit_rate AS STRING) AS detail
FROM lines_enriched l
JOIN po_lines pl ON pl.po_id = l.po_id AND pl.item_code = l.ref_id
CROSS JOIN params p
WHERE l.line_type IN ('SOV', 'CO') AND l.qty IS NOT NULL
  AND (l.unit_rate - pl.contract_unit_rate) > p.rate_round_usd;  -- no % tolerance: rounding allowance only

-- ---------------------------------------------------------------------
-- RULE 2  QTY_OVER_RECEIPT  (receipt leg): billed qty vs field-verified qty
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_qty AS
SELECT l.invoice_id, l.line_no, 'QTY_OVER_RECEIPT' AS rule,
       CASE WHEN 100.0 * (l.qty - g.verified_qty) / g.verified_qty > p.qty_hold_pct THEN 'HOLD' ELSE 'REVIEW' END AS severity,
       ROUND(100.0 * (l.qty - g.verified_qty) / g.verified_qty, 2) AS variance,
       ROUND((l.qty - g.verified_qty) * l.unit_rate, 2) AS amount_at_risk,
       'Billed ' || CAST(l.qty AS STRING) || ' vs verified ' || CAST(g.verified_qty AS STRING) || ' ' || l.unit AS detail
FROM lines_enriched l
JOIN goods_receipts g ON g.grn_id = l.grn_id
CROSS JOIN params p
WHERE l.line_type = 'SOV' AND l.qty IS NOT NULL AND g.verified_qty > 0
  AND 100.0 * (l.qty - g.verified_qty) / g.verified_qty > p.qty_review_pct;

-- Billed work with no receipt on file at all
CREATE OR REPLACE VIEW r_no_receipt AS
SELECT l.invoice_id, l.line_no, 'NO_RECEIPT' AS rule, 'HOLD' AS severity,
       NULL AS variance, l.amount AS amount_at_risk,
       'No field verification on file for ' || l.ref_id AS detail
FROM lines_enriched l
LEFT JOIN goods_receipts g ON g.grn_id = l.grn_id
WHERE l.line_type = 'SOV' AND g.grn_id IS NULL;

-- ---------------------------------------------------------------------
-- RULE 3  PCT_COMPLETE_OVERBILL: lump-sum % complete billed vs verified
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_pct AS
SELECT l.invoice_id, l.line_no, 'PCT_COMPLETE_OVERBILL' AS rule,
       CASE WHEN l.pct_complete_cum - g.verified_pct_complete_cum > p.pct_hold_pts THEN 'HOLD' ELSE 'REVIEW' END AS severity,
       ROUND(l.pct_complete_cum - g.verified_pct_complete_cum, 2) AS variance,
       ROUND(pl.scheduled_value * (l.pct_complete_cum - g.verified_pct_complete_cum) / 100.0, 2) AS amount_at_risk,
       'Billed ' || CAST(l.pct_complete_cum AS STRING) || '% vs verified ' || CAST(g.verified_pct_complete_cum AS STRING) || '%' AS detail
FROM lines_enriched l
JOIN goods_receipts g ON g.grn_id = l.grn_id
JOIN po_lines pl ON pl.po_id = l.po_id AND pl.item_code = l.ref_id
CROSS JOIN params p
WHERE l.unit = 'LS' AND l.line_type = 'SOV'
  AND l.pct_complete_cum - g.verified_pct_complete_cum > p.pct_review_pts;

-- ---------------------------------------------------------------------
-- RULE 4  UNAPPROVED_CO (pre-approval gate): CO billed without approval
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_co AS
SELECT l.invoice_id, l.line_no, 'UNAPPROVED_CO' AS rule, 'HOLD' AS severity,
       NULL AS variance, l.amount AS amount_at_risk,
       l.ref_id || ' status: ' || COALESCE(co.status, 'NOT IN CO LOG') AS detail
FROM lines_enriched l
LEFT JOIN change_orders co ON co.co_id = l.ref_id
WHERE l.line_type = 'CO'
  AND (co.co_id IS NULL OR co.status <> 'APPROVED' OR co.approval_period > l.period);

-- ---------------------------------------------------------------------
-- RULE 5  TM_OVER_APPROVED (pre-approval gate): T&M hours vs approved hours
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_tm AS
SELECT l.invoice_id, l.line_no,
       CASE WHEN t.approval_id IS NULL THEN 'TM_NOT_PREAPPROVED' ELSE 'TM_OVER_APPROVED' END AS rule,
       CASE WHEN t.approval_id IS NULL THEN 'HOLD'
            WHEN 100.0 * (l.qty - t.approved_hours) / t.approved_hours > p.tm_hold_pct THEN 'HOLD'
            ELSE 'REVIEW' END AS severity,
       CASE WHEN t.approval_id IS NULL THEN NULL
            ELSE ROUND(100.0 * (l.qty - t.approved_hours) / t.approved_hours, 2) END AS variance,
       CASE WHEN t.approval_id IS NULL THEN l.amount
            ELSE ROUND((l.qty - t.approved_hours) * l.unit_rate, 2) END AS amount_at_risk,
       'Billed ' || CAST(l.qty AS STRING) || ' hrs vs approved ' || COALESCE(CAST(t.approved_hours AS STRING), 'none') AS detail
FROM lines_enriched l
LEFT JOIN tm_preapprovals t ON t.approval_id = l.ref_id
CROSS JOIN params p
WHERE l.line_type = 'TM'
  AND (t.approval_id IS NULL OR 100.0 * (l.qty - t.approved_hours) / t.approved_hours > p.tm_review_pct);

-- ---------------------------------------------------------------------
-- RULE 6  PER_DIEM_OVER_POLICY
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_per_diem AS
SELECT l.invoice_id, l.line_no, 'PER_DIEM_OVER_POLICY' AS rule, 'HOLD' AS severity,
       ROUND(100.0 * (l.unit_rate - p.per_diem_cap) / p.per_diem_cap, 2) AS variance,
       ROUND((l.unit_rate - p.per_diem_cap) * l.qty, 2) AS amount_at_risk,
       'Billed $' || CAST(l.unit_rate AS STRING) || '/worker-day vs cap $' || CAST(p.per_diem_cap AS STRING) AS detail
FROM lines_enriched l CROSS JOIN params p
WHERE l.line_type = 'PER_DIEM' AND l.unit_rate > p.per_diem_cap + 0.005;

-- ---------------------------------------------------------------------
-- RULE 7  EXTENSION_ERROR: amount <> qty x rate (or LS % math)
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_extension AS
SELECT l.invoice_id, l.line_no, 'EXTENSION_ERROR' AS rule, 'HOLD' AS severity,
       NULL AS variance,
       ROUND(l.amount - CASE WHEN l.qty IS NOT NULL THEN l.qty * l.unit_rate
                             ELSE pl.scheduled_value * (l.pct_complete_cum - l.pct_complete_prev) / 100.0 END, 2) AS amount_at_risk,
       'Line amount does not equal quantity x rate' AS detail
FROM lines_enriched l
LEFT JOIN po_lines pl ON pl.po_id = l.po_id AND pl.item_code = l.ref_id
CROSS JOIN params p
WHERE ABS(l.amount - CASE WHEN l.qty IS NOT NULL THEN l.qty * l.unit_rate
                          ELSE pl.scheduled_value * (l.pct_complete_cum - l.pct_complete_prev) / 100.0 END) > p.money_tol;

-- ---------------------------------------------------------------------
-- RULE 8  RETAINAGE_ERROR: withheld vs contract retainage on SOV + CO work
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_retainage AS
WITH base AS (
  SELECT i.invoice_id, i.retainage_withheld, po.retainage_rate,
         SUM(CASE WHEN il.line_type IN ('SOV', 'CO') THEN il.amount ELSE 0 END) AS ret_base
  FROM invoices i
  JOIN purchase_orders po ON po.po_id = i.po_id
  JOIN invoice_lines il ON il.invoice_id = i.invoice_id
  GROUP BY i.invoice_id, i.retainage_withheld, po.retainage_rate
)
SELECT b.invoice_id, NULL AS line_no, 'RETAINAGE_ERROR' AS rule, 'HOLD' AS severity,
       ROUND(b.retainage_withheld - b.ret_base * b.retainage_rate, 2) AS variance,
       ROUND(b.ret_base * b.retainage_rate - b.retainage_withheld, 2) AS amount_at_risk,
       'Withheld ' || CAST(b.retainage_withheld AS STRING) || ' vs required ' || CAST(ROUND(b.ret_base * b.retainage_rate, 2) AS STRING) AS detail
FROM base b CROSS JOIN params p
WHERE ABS(b.retainage_withheld - b.ret_base * b.retainage_rate) > p.money_tol;

-- ---------------------------------------------------------------------
-- RULE 9  DUPLICATE_INVOICE: same PO + period + amount as an earlier invoice
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_duplicate AS
-- Fuzzy duplicate check. Exact invoice-number matching misses resubmissions such as
-- "INV-5521" vs "5521-R", so a later bill is flagged when, on the same PO and billing month,
--   (a) the amount is within 2% of an earlier bill, or
--   (b) the normalized invoice numbers match (case, spaces, dashes and suffixes like R/REV/A ignored);
-- and, across different POs of the same contractor, when the amount matches to the dollar within 45 days.
WITH n AS (
  SELECT i.*, REGEXP_REPLACE(REGEXP_REPLACE(UPPER(i.invoice_number), '[^A-Z0-9]', ''), '(REV|COPY|R|A)$', '') AS num_key,
         CAST(i.invoice_date AS DATE) AS d
  FROM invoices i
), pairs AS (
  SELECT later.invoice_id, later.net_amount, earlier.invoice_id AS earlier_id, earlier.invoice_number AS earlier_no,
         CASE WHEN later.po_id = earlier.po_id AND later.period = earlier.period AND later.num_key = earlier.num_key THEN 'same invoice number (normalized)'
              WHEN later.po_id = earlier.po_id AND later.period = earlier.period
                   AND ABS(earlier.gross_amount - later.gross_amount) <= 0.02 * GREATEST(ABS(earlier.gross_amount), 1) THEN 'same PO and month, amount within 2%'
              WHEN later.po_id <> earlier.po_id AND later.prime_id = earlier.prime_id
                   AND ABS(earlier.gross_amount - later.gross_amount) <= 1.00
                   AND ABS(DATE_DIFF(later.d, earlier.d, DAY)) <= 45 THEN 'same contractor and amount on another PO'
         END AS why
  FROM n later JOIN n earlier
    ON earlier.invoice_id <> later.invoice_id
   AND (earlier.d < later.d OR (earlier.d = later.d AND earlier.invoice_id < later.invoice_id))
)
SELECT invoice_id, NULL AS line_no, 'DUPLICATE_INVOICE' AS rule, 'HOLD' AS severity,
       NULL AS variance, MAX(net_amount) AS amount_at_risk,
       'Possible duplicate of ' || MIN(earlier_id) || ' (' || MIN(earlier_no) || '): ' || MIN(why) AS detail
FROM pairs WHERE why IS NOT NULL
GROUP BY invoice_id;

-- ---------------------------------------------------------------------
-- RULE 10 HEADER_MISMATCH: header totals disagree with lines
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW r_header AS
WITH s AS (SELECT invoice_id, SUM(amount) AS line_total FROM invoice_lines GROUP BY invoice_id)
SELECT i.invoice_id, NULL AS line_no, 'HEADER_MISMATCH' AS rule, 'HOLD' AS severity,
       ROUND(i.gross_amount - s.line_total, 2) AS variance,
       ROUND(i.gross_amount - s.line_total, 2) AS amount_at_risk,
       'Header gross/net does not reconcile to lines' AS detail
FROM invoices i JOIN s ON s.invoice_id = i.invoice_id CROSS JOIN params p
WHERE ABS(i.gross_amount - s.line_total) > p.money_tol
   OR ABS(i.net_amount - (i.gross_amount - i.retainage_withheld)) > p.money_tol;

-- ---------------------------------------------------------------------
-- All exceptions
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE exceptions AS
SELECT * FROM r_rate      UNION ALL SELECT * FROM r_qty       UNION ALL
SELECT * FROM r_no_receipt UNION ALL SELECT * FROM r_pct      UNION ALL
SELECT * FROM r_co        UNION ALL SELECT * FROM r_tm        UNION ALL
SELECT * FROM r_per_diem  UNION ALL SELECT * FROM r_extension UNION ALL
SELECT * FROM r_retainage UNION ALL SELECT * FROM r_duplicate UNION ALL
SELECT * FROM r_header;

-- ---------------------------------------------------------------------
-- Governance gate per invoice: worst severity wins
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE invoice_gates AS
SELECT i.invoice_id, i.invoice_number, i.project_id, i.prime_id, i.period, i.invoice_date,
       i.gross_amount, i.retainage_withheld, i.net_amount,
       CASE WHEN SUM(CASE WHEN e.severity = 'HOLD' THEN 1 ELSE 0 END) > 0 THEN 'HOLD'
            WHEN COUNT(e.rule) > 0 THEN 'REVIEW' ELSE 'PASS' END AS gate,
       COUNT(e.rule) AS n_exceptions,
       ROUND(COALESCE(SUM(CASE WHEN e.amount_at_risk > 0 THEN e.amount_at_risk END), 0), 2) AS amount_at_risk,
       STRING_AGG(DISTINCT e.rule, '; ') AS rules_triggered
FROM invoices i
LEFT JOIN exceptions e ON e.invoice_id = i.invoice_id
GROUP BY i.invoice_id, i.invoice_number, i.project_id, i.prime_id, i.period, i.invoice_date,
         i.gross_amount, i.retainage_withheld, i.net_amount;
