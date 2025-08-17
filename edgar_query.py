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
logger = logging.getLogger("edgar_spcl_sit")

load_dotenv()

# ========== Keywords and classification function ==========
# Keywords for special situations detection
KEYWORDS = [
    'merger', 'acquisition', 'acquires', 'acquired', 'to be acquired', 'combination',
    'tender offer', 'going private', 'go-private', 'take private',
    'spin-off', 'spinoff', 'split-off', 'carve-out',
    'restructuring', 'recapitalization', 'rights offering',
    'asset sale', 'divestiture', 'sell division', 'disposition',
    'chapter 11', 'bankruptcy', 'emerges from chapter',
    'activist', '13d', '13d/a', 'strategic review',
    'special dividend', 'buyback', 'share repurchase', 'scheme of arrangement',
    '13e-3', 'sc to-t', 'sc to-i'
]

def classify_(text: str) -> str:
    """
    Classify a text snippet into deal/action categories based on keywords.
    """
    if not text:
        return 'Other'

    t = text.lower()

    # Form-type hints from SEC titles/links
    if '13e-3' in t:
        return 'Going-Private (13E-3)'
    if 'sc to-t' in t:
        return 'Tender Offer (TO-T)'
    if 'sc to-i' in t:
        return 'Issuer Tender (TO-I)'

    if ('tender offer' in t or 'going private' in t or 'go-private' in t or 'take private' in t):
        return 'Tender/Going-Private'
    if ('spin-off' in t or 'spinoff' in t or 'split-off' in t or 'carve-out' in t):
        return 'Spin-off'
    if ('merger' in t or 'acquisition' in t or 'acquires' in t or 'to be acquired' in t or 'combination' in t):
        return 'M&A'
    if ('restructuring' in t or 'recapitalization' in t or 'rights offering' in t):
        return 'Restructuring/Recap'
    if ('asset sale' in t or 'divestiture' in t or 'sell division' in t or 'disposition' in t):
        return 'Asset Sale'
    if ('chapter 11' in t or 'bankruptcy' in t or 'emerges from chapter' in t):
        return 'Bankruptcy'
    if '13d' in t:
        return 'Activist/13D'
    if 'strategic review' in t:
        return 'Strategic Review'
    if 'special dividend' in t:
        return 'Special Dividend'
    if 'buyback' in t or 'share repurchase' in t:
        return 'Buyback'

    return 'Other'

# ========== Core search ==========
def find_special_situations(
    date_str: str,
    forms: List[str],
    include_exhibits: bool = True,
    max_filings: Optional[int] = None,
    log: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Scan all filings of specified forms on `date_str` (YYYY-MM-DD) for 'merger'/'acquisition'.
    Set max_filings to limit how many filings are scanned (useful for testing).
    """
    log = log or logger
    log.info(
        "Starting special situations scan for date=%s (forms=%s, include_exhibits=%s, max_filings=%s)",
        date_str, forms, include_exhibits, max_filings
    )

    keyword_pats = [re.compile(r"\b" + keyword + r"\b", re.IGNORECASE) for keyword in KEYWORDS]

    def has_keywords(text: Optional[str]) -> bool:
        if not text:
            return False
        return any(pat.search(text) is not None for pat in keyword_pats)

    rows: List[dict] = []
    for form in forms:
        gen = get_filings(form=form, filing_date=date_str)
        if max_filings is not None:
            gen = islice(gen, max_filings)
            log.info("Limiting scan to the first %d filings for testing.", max_filings)

        filings = list(gen)
        log.info("Fetched %d %s filings for %s", len(filings), form, date_str)

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

            if where:
                rows.append({
                    "company": f.company,
                    "cik": f.cik,
                    "filing_date": str(f.filing_date),
                    "accession_no": f.accession_no,
                    "form_type": form,
                    "classification": classify_(f.text()),
                    "where_found": "; ".join(sorted(set(where))),
                    "link": getattr(f, "url", None),
                })

            if idx % 10 == 0:
                log.info("Scanned %d/%d filings...", idx, len(filings))

    if not rows:
        # Return empty DataFrame with proper columns
        df = pd.DataFrame(columns=[
            "company", "cik", "filing_date", "accession_no", 
            "form_type", "classification", "where_found", "link"
        ])
    else:
        df = pd.DataFrame(rows).sort_values(["company", "filing_date"]).reset_index(drop=True)
    
    log.info("Found %d matching filings for %s", len(df), date_str)
    return df

def find_single_form_situations(
    date_str: str,
    form_type: str,
    include_exhibits: bool = True,
    max_filings: Optional[int] = None,
    log: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Convenience function to scan a single form type for special situations.
    Wrapper around find_special_situations for easier testing.
    
    Args:
        date_str: Filing date in YYYY-MM-DD format
        form_type: Single SEC form type (e.g., "8-K", "SC 13D", "SC 13E3", "SC TO-T")
        include_exhibits: Whether to search exhibits/attachments
        max_filings: Limit number of filings (useful for testing)
        log: Logger instance
    """
    return find_special_situations(
        date_str=date_str,
        forms=[form_type],
        include_exhibits=include_exhibits,
        max_filings=max_filings,
        log=log
    )

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
    date_to_scan = "2025-08-15"  # YYYY-MM-DD

    # For testing single form types, uncomment one of these:
    # hits = find_single_form_situations(date_to_scan, "8-K", max_filings=5, log=logger)
    #hits = find_single_form_situations(date_to_scan, "SCHEDULE 13D/A", max_filings=5, log=logger)
    # hits = find_single_form_situations(date_to_scan, "SC 13E3", max_filings=5, log=logger)
    # hits = find_single_form_situations(date_to_scan, "SC TO-T", max_filings=5, log=logger)

    # Run the scan using the new special situations function (all forms)
    hits = find_special_situations(
        date_str=date_to_scan,
        forms=["8-K", "SCHEDULE 13D", "SCHEDULE 13D/A", "SC 13E3", "SC TO-I", "SC TO-T"],
        include_exhibits=True,
        max_filings=None,
        log=logger
    )

    # Display results summary
    if not hits.empty:
        print(f"\nFound {len(hits)} special situations:")
        print(hits[['company', 'form_type', 'classification', 'where_found']].to_string(index=False))
        print(f"\nClassification breakdown:")
        print(hits['classification'].value_counts().to_string())
    else:
        print("No special situations found.")

    # Append to Google Sheet (replace with your info)
    append_df_to_gsheet(
        df=hits,
        spreadsheet_key=os.getenv("SPREADSHEET_KEY"),
        worksheet_name=os.getenv("WORKSHEET_NAME"),
        service_account_json=os.getenv("SERVICE_ACCOUNT_JSON"),
        log=logger
    )
