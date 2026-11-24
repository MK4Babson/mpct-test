/**
 * Control Tower: locked contractor submission sheet (Google Apps Script).
 * Extensions -> Apps Script -> paste -> run setup() once -> reload the sheet.
 * - Protects row 1 (headers) on every tab so column names cannot change.
 * - Adds a "Control Tower" menu: "Submit this tab" freezes the tab (whole-sheet protection),
 *   computes a SHA-256 hash of its contents and logs it on a _submissions tab.
 * The app reads each tab as CSV and recomputes the hash, so any edit after submission is detectable.
 */
function setup() {
  const ss = SpreadsheetApp.getActive();
  ss.getSheets().forEach(sh => {
    if (sh.getName().startsWith('_')) return;
    const p = sh.getRange(1, 1, 1, sh.getMaxColumns()).protect().setDescription('Headers locked by Control Tower');
    p.removeEditors(p.getEditors());
    if (p.canDomainEdit()) p.setDomainEdit(false);
  });
  if (!ss.getSheetByName('_submissions')) {
    ss.insertSheet('_submissions').appendRow(['tab', 'rows', 'sha256', 'submitted_at', 'submitted_by']);
  }
}

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Control Tower').addItem('Submit this tab', 'submitTab').addToUi();
}

function submitTab() {
  const ss = SpreadsheetApp.getActive();
  const sh = ss.getActiveSheet();
  const values = sh.getDataRange().getDisplayValues();
  const csv = values.map(r => r.map(v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v).join(',')).join('\n') + '\n';
  const digest = Utilities.computeDigest(Utilities.DigestAlgorithm.SHA_256, csv, Utilities.Charset.UTF_8)
    .map(b => ('0' + (b & 0xff).toString(16)).slice(-2)).join('');
  const p = sh.protect().setDescription('Submitted ' + new Date().toISOString());
  p.removeEditors(p.getEditors());
  if (p.canDomainEdit()) p.setDomainEdit(false);
  ss.getSheetByName('_submissions').appendRow([sh.getName(), values.length - 1, digest, new Date().toISOString(),
                                                Session.getActiveUser().getEmail()]);
  SpreadsheetApp.getUi().alert('Submitted and locked. Receipt hash: ' + digest);
}
