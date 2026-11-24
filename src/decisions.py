"""
Issue lifecycle and decision record (SQLite).

Every finding becomes an issue. A person moves it through:
    Detected -> Assigned -> Investigating -> Decided -> Reconciled -> Closed
Every change is an append-only event (who, when, what, from, to, note), so any past decision can be read back.

Four money measures, never added together:
    flagged               everything the checks found
    under review          flagged amounts on issues not yet decided (or decided "Dispute"/"Escalate")
    confirmed adjustment  amounts a reviewer decided to adjust
    money saved           confirmed adjustments already applied to a payment (Reconciled or Closed)

Demo storage: a SQLite file in the app folder. On Streamlit Community Cloud it resets when the app restarts;
download the decision log to keep it. A real deployment would use a managed database with sign-in.
"""
import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = Path(os.environ.get("MPCT_DECISIONS_DB", ROOT / "outputs" / "decisions.sqlite"))

STATUSES = ["Detected", "Assigned", "Investigating", "Decided", "Reconciled", "Closed"]
DECISIONS = ["Adjust (pay less)", "Certify as billed", "Dispute", "Escalate"]
REASONS = ["Above contract rate", "Not verified on site", "Not approved", "Missing document", "Arithmetic or ledger error",
           "Not reimbursable under the contract", "Evidence supplied: flag cleared", "Contract allows it", "Other"]
OPEN = {"Detected", "Assigned", "Investigating"}


def _con():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS issues (issue_id TEXT PRIMARY KEY, program TEXT, source TEXT, invoice_id TEXT,
                   line_no TEXT, rule TEXT, contractor TEXT, period TEXT, flagged REAL, recommended TEXT, detail TEXT,
                   status TEXT, owner TEXT, due TEXT, decision TEXT, confirmed REAL, reason TEXT, override INTEGER,
                   detected_at TEXT, updated_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, issue_id TEXT, ts TEXT,
                   user TEXT, action TEXT, from_status TEXT, to_status TEXT, note TEXT)""")
    return con


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def issue_id(program, source, invoice_id, line_no, rule):
    key = f"{program}|{source}|{invoice_id}|{line_no}|{rule}"
    return "ISS-" + hashlib.sha1(key.encode()).hexdigest()[:8].upper()


def sync(program, findings):
    """Add new findings as Detected issues. Existing issues keep their status and history."""
    con = _con()
    have = {r[0] for r in con.execute("SELECT issue_id FROM issues WHERE program=?", (program,))}
    new = 0
    for r in findings.itertuples():
        ln = "" if pd.isna(r.line_no) else str(int(float(r.line_no))) if str(r.line_no).replace(".", "", 1).isdigit() else str(r.line_no)
        iid = issue_id(program, r.source, r.invoice_id, ln, r.rule)
        if iid in have:
            continue
        con.execute("INSERT INTO issues VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (iid, program, r.source, str(r.invoice_id), ln, r.rule, getattr(r, "contractor", ""), getattr(r, "period", ""),
                     float(r.amount_at_risk or 0), r.severity, r.detail, "Detected", "", "", "", None, "", 0, _now(), _now()))
        con.execute("INSERT INTO events (issue_id, ts, user, action, from_status, to_status, note) VALUES (?,?,?,?,?,?,?)",
                    (iid, _now(), "system", "Detected by rule", "", "Detected", r.detail))
        have.add(iid)
        new += 1
    con.commit()
    con.close()
    return new


def issues(program):
    con = _con()
    df = pd.read_sql_query("SELECT * FROM issues WHERE program=?", con, params=(program,))
    con.close()
    return df


def history(iid):
    con = _con()
    df = pd.read_sql_query("SELECT ts, user, action, from_status, to_status, note FROM events WHERE issue_id=? ORDER BY event_id",
                           con, params=(iid,))
    con.close()
    return df


def events(program):
    con = _con()
    df = pd.read_sql_query("""SELECT e.*, i.invoice_id, i.rule FROM events e JOIN issues i ON i.issue_id=e.issue_id
                              WHERE i.program=? ORDER BY e.event_id""", con, params=(program,))
    con.close()
    return df


def act(iid, user, to_status, note, owner=None, due=None, decision=None, confirmed=None, reason=None):
    """Record one action. Rules: a name and a note are always required; a decision needs a reason;
    certifying a recommended HOLD as billed is an override and needs a note of at least 15 characters."""
    if not (user and user.strip()):
        raise ValueError("Enter your name: every action is recorded against a person.")
    if not (note and note.strip()):
        raise ValueError("Add a note saying why.")
    con = _con()
    row = con.execute("SELECT status, recommended, flagged FROM issues WHERE issue_id=?", (iid,)).fetchone()
    if row is None:
        con.close()
        raise ValueError("Issue not found.")
    cur, rec, flagged = row
    override = 0
    if to_status == "Decided":
        if not decision or not reason:
            con.close()
            raise ValueError("A decision needs a decision type and a reason.")
        if decision.startswith("Certify") and rec == "HOLD":
            override = 1
            if len(note.strip()) < 15:
                con.close()
                raise ValueError("Certifying a recommended hold is an override: explain it in at least 15 characters.")
        if decision.startswith("Adjust"):
            confirmed = float(flagged if confirmed is None else confirmed)
        else:
            confirmed = 0.0
    sets = {"status": to_status, "updated_at": _now()}
    if owner is not None:
        sets["owner"] = owner
    if due is not None:
        sets["due"] = str(due)
    if to_status == "Decided":
        sets.update(decision=decision, confirmed=confirmed, reason=reason, override=override)
    if to_status == "Investigating" and cur in ("Decided", "Reconciled", "Closed"):
        sets.update(decision="", confirmed=None, reason="", override=0)
    con.execute(f"UPDATE issues SET {', '.join(k + '=?' for k in sets)} WHERE issue_id=?", (*sets.values(), iid))
    action = {"Assigned": "Assigned", "Investigating": "Reopened" if cur in ("Decided", "Reconciled", "Closed") else "Investigation started",
              "Decided": f"Decided: {decision}" + (" (override)" if override else ""), "Reconciled": "Applied to payment",
              "Closed": "Closed"}.get(to_status, to_status)
    detail = note + (f" | owner {owner}" if owner else "") + (f" | due {due}" if due else "") + \
        (f" | reason {reason} | adjustment ${confirmed:,.2f}" if to_status == "Decided" else "")
    con.execute("INSERT INTO events (issue_id, ts, user, action, from_status, to_status, note) VALUES (?,?,?,?,?,?,?)",
                (iid, _now(), user.strip(), action, cur, to_status, detail))
    con.commit()
    con.close()


def next_statuses(cur):
    i = STATUSES.index(cur)
    nxt = STATUSES[i + 1:i + 2]
    if cur in ("Detected", "Assigned"):
        nxt = STATUSES[i + 1:4]
    if cur in ("Decided", "Reconciled", "Closed"):
        nxt = nxt + ["Investigating"]
    return nxt


def measures(program, review_hours=0.0, hourly_cost=0.0):
    df = issues(program)
    if df.empty:
        return {"flagged": 0.0, "under_review": 0.0, "confirmed": 0.0, "saved": 0.0, "net": 0.0, "decided": 0, "total": 0,
                "overrides": 0}
    open_ = df.status.isin(OPEN) | df.decision.isin(["Dispute", "Escalate"])
    conf = df.confirmed.fillna(0)
    saved = conf[df.status.isin(["Reconciled", "Closed"])].sum()
    return {"flagged": float(df.flagged.sum()), "under_review": float(df.flagged[open_].sum()),
            "confirmed": float(conf.sum()), "saved": float(saved), "net": float(saved - review_hours * hourly_cost),
            "decided": int((~df.status.isin(OPEN)).sum()), "total": len(df), "overrides": int(df.override.fillna(0).sum())}


def reset(program):
    con = _con()
    ids = [r[0] for r in con.execute("SELECT issue_id FROM issues WHERE program=?", (program,))]
    con.executemany("DELETE FROM events WHERE issue_id=?", [(i,) for i in ids])
    con.execute("DELETE FROM issues WHERE program=?", (program,))
    con.commit()
    con.close()


def load_demo(program, seed=11):
    """Fill a realistic set of reviewer decisions (labelled as demo) so the four measures and history can be seen."""
    import numpy as np
    rng = np.random.default_rng(seed)
    df = issues(program)
    df = df[df.status == "Detected"].sort_values("flagged", ascending=False)
    people = ["A. Rivera (cost engineer)", "J. Chen (commercial manager)", "S. Patel (AP lead)"]
    base = datetime.now(timezone.utc).date()
    for k, r in enumerate(df.itertuples()):
        u = people[k % 3]
        due = base + timedelta(days=int(rng.integers(3, 15)))
        act(r.issue_id, "demo: " + u, "Assigned", "Demo decision: assigned for review.", owner=u, due=due)
        roll = rng.random()
        if roll < 0.15:
            continue
        act(r.issue_id, "demo: " + u, "Investigating", "Demo decision: requested backup from the contractor.")
        if roll < 0.30:
            continue
        if r.rule in ("QUANTITY_OVERRUN",) or roll > 0.92:
            act(r.issue_id, "demo: " + u, "Decided", "Demo decision: contractor supplied evidence; contract allows it.",
                decision="Certify as billed", reason="Evidence supplied: flag cleared")
        elif roll > 0.86:
            act(r.issue_id, "demo: " + u, "Decided", "Demo decision: contractor disputes; sent to the commercial manager.",
                decision="Dispute", reason="Other")
            continue
        else:
            share = float(rng.choice([1.0, 1.0, 0.8, 0.6]))
            act(r.issue_id, "demo: " + u, "Decided", "Demo decision: adjustment agreed with the contractor.",
                decision="Adjust (pay less)", confirmed=round(r.flagged * share, 2), reason="Not verified on site"
                if "QTY" in r.rule or "PCT" in r.rule else "Not approved" if "CO" in r.rule else "Above contract rate"
                if "RATE" in r.rule else "Missing document" if r.rule in ("LIEN_WAIVER_MISSING", "STORED_MATERIALS_UNSUPPORTED")
                else "Arithmetic or ledger error")
        if rng.random() < 0.7:
            act(r.issue_id, "demo: " + u, "Reconciled", "Demo decision: adjustment applied to the next payment certificate.")
            if rng.random() < 0.5:
                act(r.issue_id, "demo: " + u, "Closed", "Demo decision: closed after payment.")
