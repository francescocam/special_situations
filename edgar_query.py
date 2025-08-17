import logging
import re
from dotenv import load_dotenv
import os
from itertools import islice
import pandas as pd
from typing import Optional, List
from edgar import set_identity, get_filings

# ========== Logging setup ==========
# You can configure this once in your app entrypoint
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("edgar_8k_ma")

load_dotenv()

# ========== Core search ==========
def find_8k_ma(
    date_str: str,
    require_both: bool = False,
    include_exhibits: bool = True,
    max_filings: Optional[int] = None,   # <-- NEW
    log: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Scan all 8-Ks filed on `date_str` (YYYY-MM-DD) for 'merger'/'acquisition'.
    Set max_filings to limit how many filings are scanned (useful for testing).
    """
    log = log or logger
    log.info(
        "Starting 8-K scan for date=%s (require_both=%s, include_exhibits=%s, max_filings=%s)",
        date_str, require_both, include_exhibits, max_filings
    )

    gen = get_filings(form="8-K", filing_date=date_str)
    if max_filings is not None:
        gen = islice(gen, max_filings)
        log.info("Limiting scan to the first %d filings for testing.", max_filings)

    filings = list(gen)
    log.info("Fetched %d 8-K filings for %s", len(filings), date_str)

    m_pat = re.compile(r"\bmergers?\b", re.IGNORECASE)
    a_pat = re.compile(r"\bacquisitions?\b", re.IGNORECASE)

    def has_keywords(text: Optional[str]) -> bool:
        if not text:
            return False
        return ((m_pat.search(text) is not None) and (a_pat.search(text) is not None)) \
            if require_both else ((m_pat.search(text) is not None) or (a_pat.search(text) is not None))

    rows: List[dict] = []
    for idx, f in enumerate(filings, 1):
        where = []

        try:
            if has_keywords(f.text()):
                where.append("primary")
        except Exception as e:
            log.warning("Primary text failed for %s (%s): %s", f.accession_no, f.company, e)

        if include_exhibits:
            try:
                for att in getattr(f, "attachments", []) or []:
                    doc = getattr(att, "document", "") or ""
                    typ = getattr(att, "type", "") or ""
                    if doc.lower().endswith((".htm", ".html", ".txt")) or typ.upper().startswith(("EX", "EX-")):
                        try:
                            if has_keywords(att.content()):
                                where.append(f"attachment:{doc or typ}")
                        except Exception as inner_e:
                            log.debug("Skip attachment %s for %s: %s", doc, f.accession_no, inner_e)
            except Exception as e:
                log.warning("Attachments failed for %s: %s", f.accession_no, e)

        try:
            eightk = f.obj()
            if getattr(eightk, "has_press_release", False):
                for pr in getattr(eightk, "press_releases", []) or []:
                    if has_keywords(getattr(pr, "content", "") or ""):
                        where.append("press_release")
        except Exception as e:
            log.debug("8-K object/press releases not available for %s: %s", f.accession_no, e)

        if where:
            rows.append({
                "company": f.company,
                "cik": f.cik,
                "filing_date": str(f.filing_date),
                "accession_no": f.accession_no,
                "items": getattr(f, "items", None),
                "where_found": "; ".join(sorted(set(where))),
                "link": getattr(f, "url", None),
            })

        if idx % 10 == 0:
            log.info("Scanned %d/%d filings...", idx, len(filings))

    df = pd.DataFrame(rows).sort_values(["company", "filing_date"]).reset_index(drop=True)
    log.info("Found %d matching filings for %s", len(df), date_str)
    return df
# ========== Google Sheets append ==========
def append_df_to_gsheet(
    df: pd.DataFrame,
    spreadsheet_key: str,
    worksheet_name: str = "8K_MA_hits",
    service_account_json: Optional[str] = None,
    log: Optional[logging.Logger] = None
) -> None:
    """
    Appends df rows to a Google Sheet (creates worksheet if needed).
    - `spreadsheet_key`: the long key in the sheet’s URL
    - `worksheet_name`: tab name to write to
    - `service_account_json`: path to your service-account JSON key
    """
    import gspread
    from google.oauth2.service_account import Credentials

    log = log or logger

    if df.empty:
        log.info("DataFrame is empty—nothing to append.")
        return

    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(service_account_json, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_key)
    try:
        ws = sh.worksheet(worksheet_name)
        log.info("Using existing worksheet: %s", worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=100, cols=max(10, len(df.columns)))
        log.info("Created worksheet: %s", worksheet_name)

    # Check if header exists (look only at first row to avoid pulling the whole sheet)
    first_row = ws.get("1:1")
    header_present = bool(first_row and any(cell.strip() for cell in first_row[0]))

    # Prepare rows
    data_rows = df.fillna("").astype(str).values.tolist()
    if not header_present:
        header = list(df.columns)
        ws.append_row(header, value_input_option="RAW")
        log.info("Wrote header with %d columns.", len(header))

    ws.append_rows(data_rows, value_input_option="RAW")
    log.info("Appended %d rows to '%s' in sheet %s.", len(data_rows), worksheet_name, spreadsheet_key)

# ========== Example usage ==========
if __name__ == "__main__":
    # Identify yourself to EDGAR (required)
    set_identity(os.getenv("EDGAR_IDENTITY"))

    # Choose the filing date to scan
    date_to_scan = "2025-08-08"  # YYYY-MM-DD

    # Run the scan
    hits = find_8k_ma(
        date_str=date_to_scan,
        require_both=False,          # True -> both 'merger' AND 'acquisition' must appear
        include_exhibits=True,
        max_filings=20,
        log=logger
    )

    # Append to Google Sheet (replace with your info)
    append_df_to_gsheet(
        df=hits,
        spreadsheet_key=os.getenv("SPREADSHEET_KEY"),
        worksheet_name=os.getenv("WORKSHEET_NAME"),
        service_account_json=os.getenv("SERVICE_ACCOUNT_JSON"),
        log=logger
    )
