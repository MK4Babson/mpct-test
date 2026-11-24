"""
Getting data in without manual CSV uploads.

  F19  Bill Reader      PDF pay application -> header + lines (pdfplumber; Gemini for scans) -> rules engine
  F20  Delivery photos  photo of a delivery note -> GPS / time check against the site (EXIF), plate + note number
                        (Tesseract OCR, or Gemini), matched to the PO line, receipts and payment status
  F22  Google Sheets    read locked contractor templates published from Google Sheets; freeze each submission with a hash
"""
import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ai

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
LINE_COLS = ["line_no", "line_type", "ref_id", "description", "unit", "qty", "unit_rate", "amount", "pct_complete_prev", "pct_complete_cum"]


# ---------------------------------------------------------------- F19: write sample PDFs
def write_bill_pdf(path, inv, lines, prime_name, project_name):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    doc = SimpleDocTemplate(str(path), pagesize=landscape(letter), leftMargin=30, rightMargin=30, topMargin=30, bottomMargin=30)
    ss = getSampleStyleSheet()
    head = [Paragraph(f"<b>{prime_name}</b>", ss["Title"]), Paragraph("APPLICATION FOR PAYMENT", ss["Heading2"]),
            Paragraph(f"Invoice number: {inv['invoice_number']}<br/>PO: {inv['po_id']}<br/>Project: {inv['project_id']} - {project_name}<br/>"
                      f"Billing period: {inv['period']}<br/>Invoice date: {inv['invoice_date']}", ss["Normal"]), Spacer(1, 10)]
    data = [["Line", "Type", "Ref", "Description", "Unit", "Qty", "Rate", "Amount", "% prev", "% cum"]]
    def fmt(x, d=2):  # keep full precision so qty x rate still equals the amount after reading back
        if pd.isna(x):
            return ""
        t = f"{x:,.4f}".rstrip("0").rstrip(".")
        return t if "." in t and len(t.split(".")[1]) >= d else f"{x:,.{d}f}" if abs(x - round(x, d)) < 1e-9 else t
    for r in lines.itertuples():
        data.append([int(r.line_no), r.line_type, r.ref_id, str(r.description)[:48], r.unit, fmt(r.qty, 1), fmt(r.unit_rate),
                     fmt(r.amount), fmt(r.pct_complete_prev, 1), fmt(r.pct_complete_cum, 1)])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde6f3")),
                           ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ALIGN", (5, 1), (-1, -1), "RIGHT")]))
    tot = [Spacer(1, 10), Paragraph(f"Gross amount: ${inv['gross_amount']:,.2f}<br/>Retainage withheld: ${inv['retainage_withheld']:,.2f}<br/>"
                                    f"<b>Net amount due: ${inv['net_amount']:,.2f}</b>", ss["Normal"])]
    doc.build(head + [t] + tot)


def build_samples(tables, gates, names, labels, folder=SAMPLES):
    folder.mkdir(parents=True, exist_ok=True)
    inv = tables["invoices"].merge(gates[["invoice_id", "gate"]], on="invoice_id")
    picks = []
    for gate in ["PASS", "HOLD"]:
        c = inv[inv.gate == gate].sort_values("gross_amount", ascending=False)
        if len(c):
            picks.append((gate, c.iloc[min(3, len(c) - 1)]))
    out = []
    for gate, r in picks:
        lines = tables["invoice_lines"][tables["invoice_lines"].invoice_id == r.invoice_id].sort_values("line_no")
        safe = re.sub(r"[^A-Za-z0-9_-]+", "-", str(r.invoice_number)).strip("-")
        p = folder / f"bill_{safe}_{gate.lower()}.pdf"
        write_bill_pdf(p, r, lines, names.get(r.prime_id, r.prime_id), labels.get(r.project_id, r.project_id))
        out.append(p)
    return out


# ---------------------------------------------------------------- F19: read a PDF
HDR = {"invoice_number": r"Invoice number:\s*(\S+)", "po_id": r"PO:\s*(\S+)", "project_id": r"Project:\s*(\S+)",
       "period": r"Billing period:\s*(\d{4}-\d{2})", "invoice_date": r"Invoice date:\s*(\d{4}-\d{2}-\d{2})",
       "gross_amount": r"Gross amount:\s*\$?([\d,\.]+)", "retainage_withheld": r"Retainage withheld:\s*\$?([\d,\.]+)",
       "net_amount": r"Net amount due:\s*\$?([\d,\.]+)"}


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("$", "")) if str(x).strip() not in ("", "None") else np.nan
    except ValueError:
        return np.nan


def read_bill_pdf(data):
    """Return (header dict, lines DataFrame, method, warnings)."""
    import pdfplumber
    warn = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        rows = []
        for p in pdf.pages:
            for t in p.extract_tables():
                for row in t:
                    if row and row[0] and str(row[0]).strip().isdigit():
                        rows.append(row[:10])
    if len(text.strip()) < 40:  # scanned image: no text layer
        if ai.available():
            return read_bill_ai(data)
        return {}, pd.DataFrame(columns=LINE_COLS), "none", ["This PDF has no text layer (a scan). Set GEMINI_API_KEY to read scans."]
    hdr = {}
    for k, rx in HDR.items():
        m = re.search(rx, text)
        hdr[k] = (m.group(1) if m else None)
    for k in ["gross_amount", "retainage_withheld", "net_amount"]:
        hdr[k] = _num(hdr[k]) if hdr[k] else np.nan
    lines = pd.DataFrame(rows, columns=LINE_COLS)
    for c in ["qty", "unit_rate", "amount", "pct_complete_prev", "pct_complete_cum"]:
        lines[c] = lines[c].map(_num)
    lines["line_no"] = lines.line_no.astype(int)
    for c in ["line_type", "ref_id", "description", "unit"]:
        lines[c] = lines[c].astype(str).str.replace("\n", " ").str.strip()
    missing = [k for k, v in hdr.items() if v is None or (isinstance(v, float) and np.isnan(v))]
    if missing:
        warn.append("Could not read: " + ", ".join(missing))
    if len(lines) and pd.notna(hdr.get("gross_amount")) and abs(lines.amount.sum() - hdr["gross_amount"]) > 1:
        warn.append(f"Lines add up to ${lines.amount.sum():,.2f} but the header says ${hdr['gross_amount']:,.2f}")
    return hdr, lines, "pdf text + table extraction", warn


def read_bill_ai(data):
    prompt = ("Extract this construction pay application as JSON with keys: invoice_number, po_id, project_id, period (YYYY-MM), "
              "invoice_date (YYYY-MM-DD), gross_amount, retainage_withheld, net_amount, and lines: a list of objects with "
              f"{', '.join(LINE_COLS)}. line_type is one of SOV, CO, TM, PER_DIEM. Use null when a value is absent. Numbers without $ or commas.")
    js, err = ai.generate_json(prompt, pdf=data)
    if err:
        return {}, pd.DataFrame(columns=LINE_COLS), "gemini", [err]
    lines = pd.DataFrame(js.pop("lines", []), columns=LINE_COLS)
    return js, lines, f"Gemini ({ai.model_name()})", ["Read by AI: check every number against the PDF before relying on it."]


def to_tables(hdr, lines, tables, invoice_id=None):
    """Insert the parsed bill into a copy of the program tables (replacing the same invoice if it exists)."""
    t = {k: v.copy() for k, v in tables.items()}
    inv = t["invoices"]
    existing = inv[inv.invoice_number == hdr.get("invoice_number")]
    iid = invoice_id or (existing.invoice_id.iloc[0] if len(existing) else f"INV-NEW-{hdr.get('invoice_number')}")
    po = t["purchase_orders"].set_index("po_id")
    if hdr.get("po_id") not in po.index:
        raise ValueError(f"PO {hdr.get('po_id')} is not in this program's purchase orders.")
    row = {"invoice_id": iid, "invoice_number": hdr["invoice_number"], "po_id": hdr["po_id"], "project_id": po.project_id[hdr["po_id"]],
           "prime_id": po.prime_id[hdr["po_id"]], "period": hdr["period"], "invoice_date": pd.to_datetime(hdr["invoice_date"]),
           "gross_amount": hdr["gross_amount"], "retainage_withheld": hdr["retainage_withheld"], "net_amount": hdr["net_amount"]}
    t["invoices"] = pd.concat([inv[inv.invoice_id != iid], pd.DataFrame([row])], ignore_index=True)
    old = t["invoice_lines"]
    l2 = lines.copy()
    l2["invoice_id"] = iid
    # link SOV lines to the field receipt for that item and period, as the contractor's system would
    grn = t["goods_receipts"]
    key = grn.set_index(["po_id", "item_code", "period"]).grn_id
    l2["grn_id"] = [key.get((hdr["po_id"], r, hdr["period"])) if lt == "SOV" else None for r, lt in zip(l2.ref_id, l2.line_type)]
    t["invoice_lines"] = pd.concat([old[old.invoice_id != iid], l2[old.columns.intersection(l2.columns)]], ignore_index=True)
    return t, iid


# ---------------------------------------------------------------- F20: delivery photos
def haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p = math.pi / 180
    h = 0.5 - math.cos((la2 - la1) * p) / 2 + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2
    return 12742 * math.asin(math.sqrt(h))


def _dms(v):
    d = int(v)
    m = int((v - d) * 60)
    s = round(((v - d) * 60 - m) * 60, 2)
    return (float(d), float(m), float(s))


def write_delivery_photo(path, fields, lat, lng, when):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (900, 620), (238, 234, 222))
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        f2 = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    d.rectangle([20, 20, 880, 600], outline=(60, 60, 60), width=3)
    d.text((40, 40), "DELIVERY NOTE / BILL OF LADING", fill=(20, 20, 20), font=f1)
    y = 120
    for k, v in fields.items():
        d.text((40, y), f"{k}: {v}", fill=(25, 25, 25), font=f2)
        y += 52
    d.text((40, 540), "Received by: ________   Signature: ________", fill=(60, 60, 60), font=f2)
    exif = Image.Exif()
    exif[0x9003] = when.strftime("%Y:%m:%d %H:%M:%S")  # DateTimeOriginal (in the Exif IFD below too)
    exif[0x0132] = when.strftime("%Y:%m:%d %H:%M:%S")
    gps = {1: "N" if lat >= 0 else "S", 2: _dms(abs(lat)), 3: "E" if lng >= 0 else "W", 4: _dms(abs(lng))}
    exif[0x8825] = gps
    img.save(path, "JPEG", quality=92, exif=exif)


def read_photo_meta(data):
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    ex = img.getexif()
    gps = ex.get_ifd(0x8825) if ex else {}
    when = ex.get(0x0132) or (ex.get_ifd(0x8769).get(0x9003) if ex else None)
    lat = lng = None
    if gps and 2 in gps and 4 in gps:
        conv = lambda t: float(t[0]) + float(t[1]) / 60 + float(t[2]) / 3600  # noqa: E731
        lat = conv(gps[2]) * (-1 if gps.get(1) == "S" else 1)
        lng = conv(gps[4]) * (-1 if gps.get(3) == "W" else 1)
    ts = datetime.strptime(when, "%Y:%m:%d %H:%M:%S") if when else None
    return {"lat": lat, "lng": lng, "taken": ts, "size": img.size}


FIELD_RX = {"vehicle": r"Vehicle(?: plate)?:\s*([A-Z0-9\- ]+)", "note_no": r"Delivery note(?: no\.?)?:\s*([A-Z0-9\-]+)",
            "po_id": r"PO:\s*([A-Z0-9\-]+)", "item_code": r"Item:\s*([A-Z0-9\-]+)", "qty": r"Quantity:\s*([\d,\.]+)",
            "supplier": r"Supplier:\s*(.+)"}


def read_photo_text(data):
    """OCR the delivery note: Tesseract if installed, else Gemini, else nothing (the user types the fields)."""
    text, method = "", None
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(io.BytesIO(data)))
        method = "Tesseract OCR"
    except Exception:
        pass
    if not text.strip() and ai.available():
        t, err = ai.generate("Transcribe all text on this delivery note exactly, one field per line.", images=[("image/jpeg", data)])
        text, method = (t or ""), (f"Gemini ({ai.model_name()})" if t else None)
    out = {}
    for k, rx in FIELD_RX.items():
        m = re.search(rx, text or "", flags=re.I)
        out[k] = m.group(1).strip() if m else ""
    return out, method, text


def match_delivery(fields, meta, tables, project_coords, radius_km=2.0):
    po = tables["purchase_orders"].set_index("po_id")
    checks = []
    pid = po.project_id.get(fields.get("po_id")) if fields.get("po_id") in po.index else None
    checks.append(("PO found", pid is not None, f"{fields.get('po_id')} → {pid}" if pid else "PO not in this program"))
    dist = None
    if pid and meta.get("lat") is not None and project_coords.get(pid):
        dist = haversine_km((meta["lat"], meta["lng"]), project_coords[pid])
        checks.append(("Photo taken on site", dist <= radius_km, f"{dist:,.1f} km from the {pid} site pin (limit {radius_km} km)"))
    else:
        checks.append(("Photo taken on site", False, "No GPS in the photo, or no site coordinates"))
    pol = tables["po_lines"]
    line = pol[(pol.po_id == fields.get("po_id")) & (pol.item_code == fields.get("item_code"))]
    checks.append(("Item is on the PO", len(line) > 0, f"{fields.get('item_code')}: {line.description.iloc[0]}" if len(line) else "item not on this PO"))
    status, billed_qty, paid = "Received, not yet billed", 0.0, None
    if len(line):
        lines = tables["invoice_lines"].merge(tables["invoices"][["invoice_id", "po_id", "period"]], on="invoice_id")
        lm = lines[(lines.po_id == fields["po_id"]) & (lines.ref_id == fields["item_code"])]
        billed_qty = float(lm.qty.sum())
        if len(lm):
            status = "Billed: payment in process"
            pay = tables.get("payments")
            if pay is not None and len(pay):
                p = pay[pay.invoice_id.isin(lm.invoice_id)]
                if len(p) and (p.status == "Paid").all():
                    status, paid = "Billed and paid", float(p.amount_paid.sum())
    return {"project_id": pid, "distance_km": dist, "checks": checks, "status": status, "billed_qty_to_date": billed_qty, "paid": paid}


def build_photo_samples(tables, coords, folder=SAMPLES):
    folder.mkdir(parents=True, exist_ok=True)
    pol = tables["po_lines"]
    po = tables["purchase_orders"].set_index("po_id")
    cand = pol[(pol.line_source == "CONTRACT") & (pol.unit != "LS")].iloc[[1, 20]]
    out = []
    for k, (r, onsite) in enumerate(zip(cand.itertuples(), [True, False])):
        pid = po.project_id[r.po_id]
        lat, lng = coords[pid]
        if not onsite:
            lat, lng = lat + 0.31, lng - 0.42  # ~50 km away: photo taken somewhere else
        fields = {"Supplier": "Permian Industrial Supply (fictional)", "Delivery note no": f"DN-{48210 + k}", "PO": r.po_id,
                  "Item": r.item_code, "Description": r.description[:34], "Quantity": f"{round(r.contract_qty * 0.08):,}",
                  "Vehicle plate": f"TX-{'KLM' if onsite else 'RVP'}-{4021 + 37 * k}"}
        p = folder / f"delivery_{'on_site' if onsite else 'off_site'}.jpg"
        write_delivery_photo(p, fields, lat, lng, datetime(2026, 8, 18 + k, 10, 42))
        out.append(p)
    return out


# ---------------------------------------------------------------- F22: Google Sheets
def sheet_csv_url(url, gid=None):
    """Turn a Google Sheets share/edit URL into its CSV export URL."""
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url or "")
    if not m:
        return None
    g = gid or (re.search(r"[#&?]gid=(\d+)", url).group(1) if re.search(r"[#&?]gid=(\d+)", url) else "0")
    return f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv&gid={g}"


def read_sheet(url, timeout=20):
    import requests
    u = sheet_csv_url(url) if "docs.google.com" in (url or "") else url
    if not u:
        return None, "Not a Google Sheets link"
    try:
        r = requests.get(u, timeout=timeout)
        if r.status_code != 200 or r.text.lstrip().startswith("<"):
            return None, f"Could not read the sheet (HTTP {r.status_code}). Share it as 'Anyone with the link can view' or publish it."
        return pd.read_csv(io.StringIO(r.text)), None
    except Exception as e:
        return None, f"Could not reach Google Sheets: {e}"


def freeze(df, table, submitter=""):
    """Receipt for a frozen submission: SHA-256 of the canonical CSV, so later edits are detectable."""
    csv = df.to_csv(index=False).encode()
    return {"table": table, "rows": len(df), "sha256": hashlib.sha256(csv).hexdigest(), "frozen_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "submitter": submitter}


def receipt_json(receipts):
    return json.dumps(receipts, indent=2)
