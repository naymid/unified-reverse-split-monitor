"""
Unified Reverse Stock Split Monitor
----------------------------------

This script combines two independent monitoring workflows into a single
solution.  It watches for forthcoming reverse stock splits published on
TipRanks and scans the U.S. Securities and Exchange Commission (SEC)
filings for disclosures of pending reverse splits.  When a candidate
split is detected, the script attempts to determine whether fractional
shares will be rounded up using a language model and then posts an
alert to Discord.

Key features inherited from the two original scripts include:

* Scraping the TipRanks upcoming stock splits calendar and filtering
  for reverse splits.
* Looking up the current share price via ``yfinance``.
* Searching the web via SerpAPI for news about fractional share
  treatment and using OpenAI's API to summarise whether fractional
  shares are rounded up or not.
* Polling the SEC's EDGAR system for newly filed 8‑K and DEF 14A
  documents that contain keywords related to reverse stock splits and
  again using OpenAI to parse the filing and extract split details.
* Deduplicating alerts across both data sources by tracking processed
  tickers and processed SEC filings in a single JSON file.
* Posting the resulting alerts to one of two Discord webhooks:
  ``WEBHOOK_URL_YES`` for splits where the fractional shares are rounded
  up and ``WEBHOOK_URL_NO`` for all others.

Environment variables
=====================

The script reads the following configuration values from environment
variables.  When deploying on Railway or any other cloud platform,
define these variables in the service settings.

``SERPAPI_KEY``
    Your SerpAPI key.  Used to perform Google searches for news
    articles about fractional share treatment.
``OPENAI_API_KEY``
    Your OpenAI API key.  Used to generate natural language responses
    analysing articles and SEC filings.
``WEBHOOK_URL_YES``
    The Discord webhook URL to post alerts for splits where fractional
    shares are rounded up.
``WEBHOOK_URL_NO``
    The Discord webhook URL to post alerts for all other splits.

You may optionally override the polling interval and the path to the
processed data file using the ``CHECK_INTERVAL`` and
``PROCESSED_FILE`` environment variables, respectively.  Defaults are
300 seconds (5 minutes) and ``processed_data.json``.

Deployment
==========

To run this script on Railway, create a new service for a Python
application and upload this file into your repository.  Set the
environment variables mentioned above within the Railway project
settings.  The service will continuously poll TipRanks and the SEC and
post Discord alerts whenever a qualifying reverse split is discovered.
"""

import os
import time
import json
import logging
import requests
try:
    import yfinance as yf  # type: ignore
except ImportError:
    # yfinance is optional; without it stock prices will be 'N/A'
    yf = None  # type: ignore
try:
    import pandas as pd  # type: ignore
except ImportError:
    # pandas is optional; without it TipRanks scraping will fail
    pd = None  # type: ignore
from bs4 import BeautifulSoup
from datetime import datetime as dt
from pathlib import Path
try:
    from pyrate_limiter import Limiter, Rate, Duration  # type: ignore
except ImportError:
    # Provide dummy rate limiter classes if pyrate_limiter is not installed.
    class Rate:
        def __init__(self, calls: int, period: float) -> None:
            self.calls = calls
            self.period = period

    class Duration:
        SECOND = 1

    class Limiter:
        def __init__(self, rate: Rate) -> None:
            self.rate = rate

        def try_acquire(self, key: str) -> None:
            # Dummy implementation does nothing; requests are unthrottled
            return

try:
    import openai  # type: ignore
except ImportError:
    # The openai package may not be installed in the current environment
    # (e.g. during static analysis or testing).  If unavailable, the
    # functions that depend on it will raise an ImportError when called.
    openai = None  # type: ignore


# ---------------------------------------------------------------------------
# Configuration via environment variables
# ---------------------------------------------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
WEBHOOK_URL_YES = os.getenv("WEBHOOK_URL_YES", "")
WEBHOOK_URL_NO = os.getenv("WEBHOOK_URL_NO", "")

# Polling interval in seconds.  Default is 300 seconds (5 minutes).
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))

# Location of the processed data file.  This JSON file stores
# previously‑processed tickers and SEC filing identifiers to avoid
# duplicate alerts.
PROCESSED_FILE = os.getenv("PROCESSED_FILE", "processed_data.json")

# SEC API constants
URL_CIK_MAPPING = "https://www.sec.gov/files/company_tickers_exchange.json"
URL_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
URL_FILING = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_num_no_dash}/{document}"
SUPPORTED_FORMS = {"8-K", "DEF 14A"}

# The SEC imposes rate limits.  Use pyrate_limiter to throttle requests.
SEC_RATE_LIMIT = Rate(10, Duration.SECOND)
limiter = Limiter(SEC_RATE_LIMIT)

# HTTP headers for SEC requests.  Include a descriptive user agent as
# required by the SEC's terms of service.  Replace the email with a
# valid contact for production use.
HEADERS = {
    "Accept-Encoding": "gzip, deflate",
    "User-Agent": os.getenv(
        "SEC_USER_AGENT",
        "ReverseSplitMonitor bot/1.0 (contact@example.com)"
    ),
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    filename="monitor_log.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Persistent storage for processed items
# ---------------------------------------------------------------------------

def load_processed_data() -> dict:
    """Load processed tickers and filings from disk.

    The processed file stores a dictionary with two keys:

    ``tickers``
        A list of ticker symbols that have already triggered an alert.
    ``filings``
        A list of unique SEC filing identifiers ("{company}_{accession}")
        that have been processed.  This avoids duplicate alerts when
        scanning filings.

    Returns a dictionary with ``tickers`` and ``filings`` keys.
    """
    if not os.path.exists(PROCESSED_FILE):
        return {"tickers": [], "filings": []}
    try:
        with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Processed file must contain a JSON object")
            data.setdefault("tickers", [])
            data.setdefault("filings", [])
            return data
    except Exception as exc:
        logging.error(f"Failed to load processed data: {exc}")
        return {"tickers": [], "filings": []}


def save_processed_data(data: dict) -> None:
    """Persist processed tickers and filings to disk."""
    try:
        with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logging.error(f"Failed to save processed data: {exc}")


# ---------------------------------------------------------------------------
# TipRanks data source
# ---------------------------------------------------------------------------

def fetch_stock_price(ticker: str) -> str:
    """Fetch the latest closing price for the given ticker.

    Returns the price as a string rounded to two decimal places, or
    ``"N/A"`` if the price cannot be retrieved.
    """
    # If yfinance is unavailable, return N/A
    if yf is None:
        return "N/A"
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.history(period="1d")
        if not stock_info.empty:
            return f"{stock_info['Close'].iloc[-1]:.2f}"
    except Exception as exc:
        logging.error(f"Failed to fetch stock price for {ticker}: {exc}")
    return "N/A"


def fetch_stock_splits() -> list:
    """Scrape the TipRanks upcoming splits calendar for reverse splits.

    Returns a list of dictionaries, each containing ``ticker``,
    ``company``, ``split_ratio``, ``split_date`` and ``price``.
    """
    url = "https://www.tipranks.com/calendars/stock-splits/upcoming"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ReverseSplitMonitor/1.0)"
    }
    retries = 5
    # If pandas is unavailable, we cannot parse the TipRanks table
    if pd is None:
        logging.error("pandas is required to parse TipRanks data; returning empty list.")
        return []
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            tables = pd.read_html(response.text)
            if not tables:
                raise ValueError("No tables found on the page")
            df = tables[0]
            df.columns = df.columns.str.strip().str.lower()
            if "type" not in df.columns:
                raise ValueError("Expected column 'type' not found in TipRanks data")
            reverse_splits = df[df["type"].str.lower() == "reverse"]
            stock_data: list[dict] = []
            for _, row in reverse_splits.iterrows():
                ticker = str(row.get("name", "")).strip().upper()
                company = str(row.get("company", "")).strip()
                ratio = str(row.get("split ratio", "")).strip()
                date = str(row.get("split date", "")).strip()
                price = fetch_stock_price(ticker)
                stock_data.append(
                    {
                        "ticker": ticker,
                        "company": company,
                        "split_ratio": ratio,
                        "split_date": date,
                        "price": price,
                    }
                )
            return stock_data
        except Exception as exc:
            logging.error(f"Error fetching stock split data (attempt {attempt+1}): {exc}")
            time.sleep(2)  # brief pause before retrying
    logging.error("Failed to fetch stock split data after multiple attempts.")
    return []


def search_with_serpapi(query: str) -> str | None:
    """Perform a Google search via SerpAPI and return the first result URL.

    ``None`` is returned if no result is found or an error occurs.
    """
    if not SERPAPI_KEY:
        logging.error("SERPAPI_KEY environment variable is not set.")
        return None
    params = {
        "engine": "google",
        "q": query,
        "hl": "en",
        "gl": "us",
        "no_cache": True,
        "api_key": SERPAPI_KEY,
    }
    try:
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        results = response.json()
        first_result = (results.get("organic_results") or [])[:1]
        if first_result:
            return first_result[0].get("link")
    except Exception as exc:
        logging.error(f"SerpAPI error: {exc}")
    return None


def fetch_article_content(url: str) -> str:
    """Download and extract the text content from a web page.

    Returns the concatenated text of all paragraph tags, or an empty
    string if the page cannot be fetched.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return " ".join(p.get_text() for p in paragraphs)
    except Exception as exc:
        logging.error(f"Failed to fetch article content from {url}: {exc}")
        return ""


def process_with_openai(company: str, split_date: str, article_text: str) -> str:
    """Use OpenAI to determine if fractional shares are rounded up.

    Given a company name, split date and the text of a news article,
    prompt the model to answer ``Yes``, ``No`` or ``Unknown``.  Returns
    the model's trimmed response.  If the ``openai`` module is not
    available or the API key is missing, ``"Unknown"`` is returned.
    """
    if openai is None or not OPENAI_API_KEY:
        logging.error("OpenAI API key not provided or openai package not available.")
        return "Unknown"
    try:
        openai.api_key = OPENAI_API_KEY
        prompt = f"""
Analyze the following article and determine if fractional shares from {company}'s reverse stock split on {split_date} are rounded up to the nearest whole share.

Article:
{article_text}

If fractional shares are rounded up, respond with 'Yes'.
If fractional shares are not rounded up, respond with 'No'.
If the information is unclear, respond with 'Unknown'.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.2,
        )
        result = response["choices"][0]["message"]["content"].strip()
        logging.info(f"OpenAI rounding response: {result}")
        return result
    except Exception as exc:
        logging.error(f"OpenAI error: {exc}")
        return "Unknown"


# ---------------------------------------------------------------------------
# SEC filings data source
# ---------------------------------------------------------------------------

def call_sec_json(uri: str) -> dict:
    """Rate‑limited call to the SEC for JSON endpoints."""
    limiter.try_acquire("sec_api")
    resp = requests.get(uri, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_sec_filing(uri: str) -> bytes:
    """Rate‑limited download of a SEC filing document."""
    limiter.try_acquire("sec_api")
    resp = requests.get(uri, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.content


def clean_filing_content(raw_content: str) -> str:
    """Strip scripts, styles and metadata from a filing and extract text."""
    soup = BeautifulSoup(raw_content, "html.parser")
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


def fetch_new_filings(keywords: list[str], processed_data: dict) -> list:
    """Fetch SEC filings that mention the supplied keywords and have not been processed.

    This function fetches today's filings for all companies listed in
    ``company_tickers_exchange.json`` and returns a list of filings
    that match the specified keywords.  Filings are skipped if their
    unique ID (company name plus accession number) or the associated
    ticker has been processed previously.

    Parameters
    ----------
    keywords : list of str
        Keywords to search for within the raw filing text.
    processed_data : dict
        Dictionary containing ``tickers`` and ``filings`` lists used
        for deduplication.

    Returns
    -------
    list of dict
        Filings that match the criteria.  Each dict contains keys
        ``company``, ``ticker``, ``accession_number``, ``form``, ``date``,
        ``content`` and ``unique_id``.
    """
    try:
        cik_mapping = call_sec_json(URL_CIK_MAPPING)
    except Exception as exc:
        logging.error(f"Failed to fetch CIK mapping: {exc}")
        return []
    new_filings = []
    current_date = dt.now().strftime("%Y-%m-%d")
    for company in cik_mapping.get("data", []):
        try:
            cik = str(company[0]).zfill(10)
            company_name = company[1]
            ticker = company[2]
            submission_uri = URL_SUBMISSIONS.format(cik=cik)
            try:
                filings_metadata = call_sec_json(submission_uri)
            except Exception as exc:
                logging.error(f"Error fetching filings for {company_name}: {exc}")
                continue
            recent_filings = filings_metadata.get("filings", {}).get("recent", {})
            accession_numbers = recent_filings.get("accessionNumber", [])
            form_types = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            primary_docs = recent_filings.get("primaryDocument", [])
            for acc_num, form_type, filing_date, doc_name in zip(
                accession_numbers, form_types, filing_dates, primary_docs
            ):
                # Only consider today's filings of supported form types
                if filing_date != current_date or form_type not in SUPPORTED_FORMS:
                    continue
                unique_id = f"{company_name}_{acc_num}"
                if unique_id in processed_data.get("filings", []) or ticker in processed_data.get("tickers", []):
                    continue
                # Download the filing and search for keywords
                filing_uri = URL_FILING.format(
                    cik=cik,
                    acc_num_no_dash=acc_num.replace("-", ""),
                    document=doc_name,
                )
                try:
                    filing_raw = download_sec_filing(filing_uri).decode("utf-8", errors="ignore")
                except Exception as exc:
                    logging.error(f"Error downloading filing {unique_id}: {exc}")
                    continue
                # Check if all keywords are present in the raw filing
                if all(keyword.lower() in filing_raw.lower() for keyword in keywords):
                    cleaned_content = clean_filing_content(filing_raw)
                    new_filings.append(
                        {
                            "company": company_name,
                            "ticker": ticker,
                            "accession_number": acc_num,
                            "form": form_type,
                            "date": filing_date,
                            "content": cleaned_content,
                            "unique_id": unique_id,
                        }
                    )
        except Exception as exc:
            logging.error(f"Unexpected error processing company {company}: {exc}")
    return new_filings


def analyze_with_openai(content: str) -> dict | None:
    """Use OpenAI to extract reverse split details from filing content.

    Returns a dictionary with keys ``company``, ``ticker``, ``split_date``,
    ``ratio`` and ``reverse_split`` ("Yes" or "No").  If the model
    cannot parse the content or the API is unavailable, returns
    ``None``.
    """
    if openai is None or not OPENAI_API_KEY:
        logging.error("OpenAI API key not provided or openai package not available.")
        return None
    prompt = f"""
Analyze the following SEC filing content and determine if a reverse stock split is occurring. 
Extract the following details if they are present:
1. Company Name
2. Ticker
3. Date of Reverse Stock Split
4. Split Ratio (e.g., 1-for-X)

Content:
{content}

Respond with a JSON object containing:
{{
    "company": "Company Name",
    "ticker": "Ticker",
    "split_date": "Month Date, Year",
    "ratio": "1 for X.XX",
    "reverse_split": "Yes" or "No"
}}
"""
    try:
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        content_str = response["choices"][0]["message"]["content"].strip()
        return json.loads(content_str)
    except Exception as exc:
        logging.error(f"OpenAI error analysing filing: {exc}")
        return None


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_discord_alert(ticker: str, company: str, ratio: str, price: str, date: str, rounding_result: str) -> None:
    """Send a formatted alert to Discord based on rounding_result.

    Messages are always posted to the ``WEBHOOK_URL_NO`` channel.  If
    ``rounding_result`` is ``"Yes"``, the alert is also posted to
    ``WEBHOOK_URL_YES``.
    """
    message = (
        f"📉 Reverse Split Alert 📈\n"
        f"**Company:** {company}\n"
        f"**Ticker:** {ticker}\n"
        f"**Ratio:** {ratio}\n"
        f"**Current Price:** {price}\n"
        f"**Split Date:** {date}\n"
        f"**Fractional Shares Rounded:** {rounding_result}"
    )
    # Always post to the 'No' webhook for visibility
    if WEBHOOK_URL_NO:
        try:
            requests.post(WEBHOOK_URL_NO, json={"content": message}, timeout=30)
        except Exception as exc:
            logging.error(f"Error sending Discord alert (NO): {exc}")
    # If rounding result is Yes, also post to the 'Yes' webhook
    if rounding_result == "Yes" and WEBHOOK_URL_YES:
        try:
            requests.post(WEBHOOK_URL_YES, json={"content": message}, timeout=30)
        except Exception as exc:
            logging.error(f"Error sending Discord alert (YES): {exc}")


# ---------------------------------------------------------------------------
# Main monitoring loop
# ---------------------------------------------------------------------------

def monitor_reverse_splits() -> None:
    """Main loop that monitors TipRanks and SEC for reverse splits."""
    processed_data = load_processed_data()
    keywords = ["reverse stock split", "rounded up"]
    while True:
        logging.debug("🔍 Starting monitoring cycle...")
        # -----------------------------------------------------------------
        # TipRanks data source
        # -----------------------------------------------------------------
        try:
            stock_data = fetch_stock_splits()
        except Exception as exc:
            logging.error(f"Error fetching stock splits: {exc}")
            stock_data = []
        for stock in stock_data:
            ticker = stock["ticker"]
            if ticker in processed_data.get("tickers", []):
                logging.debug(f"⏭️ Skipping already processed ticker from TipRanks: {ticker}")
                continue
            company = stock["company"]
            ratio = stock["split_ratio"]
            price = stock["price"]
            date = stock["split_date"]
            query = f"{company} reverse stock split fractional shares"
            article_url = search_with_serpapi(query)
            if not article_url:
                logging.warning(f"⚠️ No article URL found for {ticker}")
                continue
            article_text = fetch_article_content(article_url)
            rounding_result = process_with_openai(company, date, article_text)
            send_discord_alert(ticker, company, ratio, price, date, rounding_result)
            # Mark ticker as processed and persist
            processed_data.setdefault("tickers", []).append(ticker)
            save_processed_data(processed_data)
        # -----------------------------------------------------------------
        # SEC data source
        # -----------------------------------------------------------------
        try:
            new_filings = fetch_new_filings(keywords, processed_data)
        except Exception as exc:
            logging.error(f"Error fetching SEC filings: {exc}")
            new_filings = []
        for filing in new_filings:
            # Skip if ticker already processed via TipRanks or another filing
            ticker = filing["ticker"]
            unique_id = filing["unique_id"]
            if ticker in processed_data.get("tickers", []) or unique_id in processed_data.get("filings", []):
                continue
            analysis = analyze_with_openai(filing["content"])
            if not analysis:
                continue
            # Only alert on filings that explicitly indicate a reverse split
            reverse_split_flag = analysis.get("reverse_split")
            if reverse_split_flag != "Yes":
                continue
            ratio = analysis.get("ratio", "N/A")
            split_date = analysis.get("split_date", "N/A")
            price = fetch_stock_price(ticker)
            # For SEC filings we interpret reverse_split == "Yes" as the
            # fractional shares being rounded up; if the filing does not
            # include such information, the model may still answer "Yes".  In
            # either case we send "Yes" to the YES channel.
            rounding_result = "Yes"
            send_discord_alert(ticker, analysis.get("company", filing["company"]), ratio, price, split_date, rounding_result)
            # Mark ticker and filing as processed and persist
            processed_data.setdefault("tickers", []).append(ticker)
            processed_data.setdefault("filings", []).append(unique_id)
            save_processed_data(processed_data)
        logging.info("✅ Monitoring cycle complete. Sleeping...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    monitor_reverse_splits()