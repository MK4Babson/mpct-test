-- Reporting views for Looker Studio (BigQuery Standard SQL)
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
