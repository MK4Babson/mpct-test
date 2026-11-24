# Bill-checking API

The same checking engine as the Streamlit app, available to other software. A contractor portal, an
accounts-payable system or a spreadsheet can send a bill and get back **PASS / REVIEW / HOLD**, the reason in
plain English, the dollars in question and the next step, in about a second.

> All bundled data is synthetic. Contractors, sites and amounts are fictional.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | About the API |
| GET | `/health` | Is it running? |
| GET | `/rules` | The 12 checks, in plain English, with default tolerances |
| GET | `/programs` | Programs available (`west-texas`, `north-koel`) |
| GET | `/programs/{program}/summary` | Bills, dollars billed, dollars in question, pass/review/hold counts |
| GET | `/programs/{program}/bills` | List bills; filter by `decision`, `contractor_id`, `project_id`, `period` |
| GET | `/programs/{program}/bills/{invoice_id}` | One bill: decision, reasons, next steps, safe-to-pay amount |
| GET | `/programs/{program}/contractors` | Contractor scorecard |
| POST | `/programs/{program}/check-bill` | **Check a new bill** against that program's contracts, approvals and past bills |
| POST | `/check` | **Bring your own data**: send every table, optionally with your own tolerances |
| GET | `/schema` | Tables and columns that `/check` expects |

Interactive documentation, where you can try every endpoint in the browser: **`/docs`**

## Example: check a new bill

```bash
curl -X POST https://YOUR-API-URL/programs/west-texas/check-bill \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_id": "API-0001", "invoice_number": "CC2-009", "po_id": "PO-WTX-C2-C",
    "period": "2026-09", "invoice_date": "2026-09-29",
    "gross_amount": 4266500, "retainage_withheld": 426650, "net_amount": 3839850,
    "lines": [
      {"line_no": 1, "line_type": "SOV", "ref_id": "C-301", "description": "Mass excavation", "unit": "CY",
       "qty": 50000, "unit_rate": 9.33, "amount": 466500, "grn_id": "GRN-API-C301"},
      {"line_no": 2, "line_type": "SOV", "ref_id": "C-302", "description": "Structural concrete - foundations",
       "unit": "CY", "qty": 8000, "unit_rate": 475.00, "amount": 3800000, "grn_id": "GRN-API-C302"}
    ],
    "field_verifications": [
      {"grn_id": "GRN-API-C301", "item_code": "C-301", "verified_qty": 48000},
      {"grn_id": "GRN-API-C302", "item_code": "C-302", "verified_qty": 8000}
    ]
  }'
```

Answer (shortened):

```json
{
  "decision": "HOLD",
  "amount_in_question": 130100.0,
  "safe_to_pay_now": 3709750.0,
  "reasons": [
    {"line_no": 1, "severity": "REVIEW", "what_is_wrong": "quantities billed above field-verified",
     "detail": "Billed 50000.0 vs verified 48000.0 CY", "amount_in_question": 18660.0,
     "next_step": "Ask the field team to re-measure the flagged items and pay verified quantities only."},
    {"line_no": 2, "severity": "HOLD", "what_is_wrong": "unit rates above contract",
     "detail": "Billed rate 475.0 vs contract 461.07", "amount_in_question": 111440.0,
     "next_step": "Pay at the contract rate and reject the difference."}
  ]
}
```

Re-sending a bill that was already paid is caught as `DUPLICATE_INVOICE`. Nothing sent to the API is saved.

## Run it on your computer

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload
# open http://127.0.0.1:8000/docs
```

## Put it online

- **Render (free):** New → Blueprint → pick this repo. `render.yaml` sets everything up. The free plan sleeps
  after 15 minutes idle, so the first call after a pause takes about 30 seconds.
- **Google Cloud Run:** `gcloud run deploy mpct-api --source . --region us-central1 --allow-unauthenticated`
  (uses the `Dockerfile`; needs a Google Cloud project with billing turned on; small use stays in the free tier).

**Optional key:** set the environment variable `MPCT_API_KEY` on the server, and the two POST endpoints will
require the header `X-API-Key: <your key>`. The read-only GET endpoints stay open.

## Files

`api/main.py` (the API) · `requirements-api.txt` · `render.yaml` · `Dockerfile` · this guide. The Streamlit app
is unchanged.
