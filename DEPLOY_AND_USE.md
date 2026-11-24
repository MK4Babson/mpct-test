# Deploy, use and test with real data: step-by-step

Four parts: **A** run it on your laptop · **B** put it on GitHub · **C** publish it as a live web app · **D** test it with real data.
The commands below are for **Windows (PowerShell)**. Mac/Linux differences are noted.

---

## Part A: Run it on your laptop (about 20 minutes)

### A1. Install the tools (one time)
| Tool | Where | Check it works |
|---|---|---|
| Python 3.12 | python.org → Downloads. **Tick "Add python.exe to PATH"** during install | `python --version` |
| Git | git-scm.com → Download for Windows (default options) | `git --version` |
| VS Code (or Antigravity) | code.visualstudio.com | open it |

### A2. Unzip and open the project
1. Unzip `multiprime-control-tower.zip` to `C:\Users\<you>\projects\multiprime-control-tower`
2. VS Code → **File → Open Folder** → pick that folder
3. **Terminal → New Terminal** (PowerShell opens inside the project folder)

### A3. Create a private Python environment and install packages
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.
You'll know it worked when your prompt starts with `(.venv)`.

### A4. Run the pipeline (proves the code works on your machine)
```powershell
python src/generate_data.py                                # creates data/raw (with planted errors)
python src/generate_data.py --clean --out data/clean_demo  # creates a clean dataset
python src/validate_input.py --data data/raw               # expect: OK
python src/run_engine.py                                   # expect: PASS 66, HOLD 48, REVIEW 3
python src/evaluate.py                                     # expect: ALL RULES 98.5% precision / 100% recall
```

### A5. Open the web app locally
```powershell
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. Stop it with `Ctrl + C` in the terminal.

---

## Part B: Put it on GitHub (about 10 minutes)

1. github.com → **+ → New repository** → name `multiprime-control-tower` → **Public** → *don't* add a README (you already have one) → **Create**.
2. In the VS Code terminal (inside the project folder):
```powershell
git init
git add .
git commit -m "Multi-Prime Commercial Control Tower: engine, evaluation, web app"
git branch -M main
git remote add origin https://github.com/manishkr2106/multiprime-control-tower.git
git push -u origin main
```
The first push opens a browser window to sign in to GitHub. Approve it.
3. Refresh the GitHub page. You should see the README with the tolerance chart.

*No-terminal alternative:* on the empty repo page click **"uploading an existing file"**, drag in everything except `.venv`, and commit.

---

## Part C: Publish the live web app (about 10 minutes, free)

1. Go to **share.streamlit.io** → **Continue with GitHub** → authorize.
2. **Create app → Deploy a public app from GitHub**
   - Repository: `manishkr2106/multiprime-control-tower`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: choose something like `mpct-manishkumar` → `https://mpct-manishkumar.streamlit.app`
   - **Advanced settings → Python version 3.12**
3. **Deploy**. The first build takes 2–5 minutes while it installs packages.
4. Open the URL and check the three data options, the tabs and the stress test.
5. Every later `git push` redeploys automatically.

**Good to know**
- Free apps go to sleep after a period with no visitors. The first visitor clicks "Yes, get this app back up" and waits about 30 seconds. Open it yourself before an interview.
- Files uploaded to the app are processed in memory for that session only. Still, **never upload confidential employer data to a public app.** Run that locally (Part A5) instead.

### Put the link to work
- Resume header: `Live project: mpct-manishkumar.streamlit.app`
- GitHub repo description: paste the URL
- Portfolio site: a project card linking to both
- LinkedIn Featured: add the app link with a one-line description

---

## Using the app

| Area | What to do | What it tells you |
|---|---|---|
| **1. Data** (sidebar) | Pick a demo or upload your CSVs | What gets checked |
| **2. Filters** (sidebar) | Pick projects, contractors and a month range | Every number and finding on the page follows these filters |
| **3. Tolerances** (sidebar, folded) | Open it and move the sliders | How strict each check is. Results update instantly |
| **KPI row** | Read | Projects in view, $ billed, first-pass rate, invoices on hold, $ at risk, pending change orders |
| **This week's decisions** | Read top to bottom; download the brief | The most important findings in plain English, each with a $ amount and a recommended action |
| **Portfolio** | Read the cards and chart | Each building: contract value, % billed, % verified complete, $ at risk, status (On track / Watch / Act now) |
| **Contractors** | Read the scorecard, trend and tables | Who bills cleanly, who is getting worse, where each contractor's risk comes from, and the same item priced differently across buildings |
| **Invoices** | Pick an invoice | Its lines and every exception with the reason |
| **Exceptions** | Filter by rule, download CSV | The work queue for the commercial team |
| **Accuracy & testing** | Read; run the stress test | Precision/recall against the answer key; plant errors in a copy and measure what is caught |

### A 3-minute interview demo script
1. *"This is a six-building program with three prime contractors: $420M billed, 48 invoices on hold, $17.5M at risk."*
2. **This week's decisions:** read the top three findings and their actions out loud.
3. Filter to one building, and show how the findings change to that building's issues.
4. **Portfolio:** *"This building has billed more than the field has verified. That's the gap I'd withhold."*
5. **Contractors:** *"This contractor's first-pass rate dropped from 76% to 42% in the last two months, so I'd call a billing-quality meeting."*
6. **Accuracy & testing:** *"On data with no answer key, I plant known errors and measure what the engine catches."*

## Part D: Test it with real data

Contractor invoices are confidential, so there's no public dataset of "real invoices with known mistakes". You test in four levels, from practice to real:

### Level 1: Practice on the clean demo (5 minutes)
App → **Demo: clean invoices** → **Stress test** → Run (40 errors, 25% tiny).
**What "working" looks like:** 0 pre-existing exceptions, about 80–95% recall, every miss marked `subtle = True`, and 0 new false alarms.
Command-line version:
```powershell
python src/inject_errors.py --data data/clean_demo --n 40 --subtle 0.25
```

### Level 2: Real public pay estimates from TxDOT (recommended, 2–3 hours)
TxDOT publishes **Contractor's Estimate Packages** for its highway contracts as PDFs: real contractors, real unit prices, quantities paid each estimate, retainage, change-order items and force-account (T&M-like) work. They sit on TxDOT's public file server under `ftp.dot.state.tx.us/pub/txdot-info/cmd/cserve/distinfo/cisrpts/`, named by contract ID (search the web for `cisrpts txdot estimate` to find recent ones). TxDOT's **Average Low Bid Unit Prices** page gives real benchmark rates for Week 2.

**Step 1: pick one contract.** Choose one with 3–6 estimates and a few change-order items. Download the PDF.

**Step 2: map it into the templates** (`templates/`, column meanings in `DATA_DICTIONARY.md`):

| TxDOT estimate package | → Our table / column |
|---|---|
| Contract ID, contractor | `purchase_orders`: `po_id` = contract ID, `prime_id` = short contractor name, `project_id` = project/CSJ |
| Retainage shown on the recap | `purchase_orders.retainage_rate` (use 0 if the estimates show $0 retainage) |
| Contract line items: item code, description, unit, **Total Bid Quantity**, **Unit Price** | `po_lines`: `item_code`, `description`, `unit`, `contract_qty`, `contract_unit_rate`, `scheduled_value` = qty × price, `line_source` = CONTRACT |
| "UNIQUE CHANGE ORDER ITEM" lines (with "Added by CO#…") | `po_lines` with `line_source` = CHANGE_ORDER, and a `change_orders` row with status APPROVED (TxDOT pays only approved changes) |
| Each **Estimate Number** | one `invoices` row: `invoice_id` = `<contract>-E<number>`, `period` from the period-ending date, `gross_amount` = total item earnings this estimate, `retainage_withheld`, `net_amount` |
| Each item with **QTY THIS ESTIMATE** ≠ 0 | one `invoice_lines` row: `line_type` SOV (or CO), `qty`, `unit_rate`, `amount` = amount paid this estimate |
| TxDOT inspectors measured that quantity | `goods_receipts`: `verified_qty` = QTY THIS ESTIMATE, `grn_id` = `<invoice_id>-<item_code>` (put the same id in the line's `grn_id`) |

Two TxDOT quirks:
- TxDOT bills lump-sum items like mobilization as a **fraction of 1 LS** (e.g. qty 0.0081). Set their `unit` to `LSQ` (not `LS`) so the engine treats them as unit-price lines.
- Negative quantities are real adjustments (corrections). Keep them.

*Shortcut: upload the PDF to me in this chat and I'll convert it into the CSVs for you. That's also a preview of the Week 3 AI invoice reader.*

**Step 3: validate.**
```powershell
python src/validate_input.py --data data/txdot_<contract>
```
Fix every ERROR. Read the WARNINGs.

**Step 4: baseline run.**
```powershell
python src/run_engine.py --data data/txdot_<contract> --out outputs/txdot_<contract>
```
These are amounts TxDOT actually paid, so expect **almost everything to PASS**. Investigate every exception. It's one of three things:
1. **A mapping mistake** (most common): fix the CSV.
2. **A real-world quirk** (rounding of 3-decimal unit prices, a price adjustment, an overrun approved by change order): adjust a tolerance or note it.
3. **A genuine anomaly**: an interesting interview story.

**Step 5: stress test on the real data.**
```powershell
python src/inject_errors.py --data data/txdot_<contract> --n 40 --subtle 0.25
```
Or upload the CSVs in the app → **Stress test**. Report the recall, missed and false-alarm numbers.

**What you can then say (honestly):** *"I mapped a real TxDOT contractor pay-estimate package into my engine. On amounts TxDOT actually paid it raised X exceptions, which I traced to Y. Planting 40 errors into that real data, it caught Z%, and every miss was below the tolerance by design."*

### Level 3: Your own past project data (optional, local only)
If you have your own spreadsheets from past projects (BOQ, RA bills, measurement book), you can map them the same way: BOQ → `po_lines`, RA bill → `invoices` + `invoice_lines`, measurement book → `goods_receipts`.
- **Only if you're allowed to use them.** Remove names, rename contractors and projects, and keep it on your laptop (Part A5), **never** in the public app or on GitHub.

### Level 4: Real PDFs, read by AI (Week 3)
An LLM reads PDF pay applications into these same CSVs, and the stress test measures extraction accuracy.

---

## How to tell the engine is working: 5 checks

| # | Check | Pass looks like |
|---|---|---|
| 1 | `validate_input.py` | `OK`, or only warnings you understand |
| 2 | Clean data → engine | (Almost) zero exceptions. Each one is explained |
| 3 | Stress test, normal-size errors | ≥ 95% caught |
| 4 | Stress test, tiny errors | Missed. They're below tolerance by design. This proves the thresholds are doing their job |
| 5 | Manual audit sample | Open all HOLDs + 10 random PASS invoices and check them by hand against the source. Agree on all |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Reinstall Python and tick "Add to PATH", or use `py` instead of `python` |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError` | Activate the venv first (`(.venv)` in the prompt), then `pip install -r requirements.txt` |
| Validator: "non-numeric value" | Remove `$`, commas and text from number columns |
| Lots of `NO_RECEIPT` holds | The line's `grn_id` doesn't match any `goods_receipts.grn_id` |
| Streamlit Cloud build fails | Open "Manage app → Logs". Usually a package version: check `requirements.txt` was pushed |
| App asleep | Click "Yes, get this app back up" and wait about 30 seconds |
