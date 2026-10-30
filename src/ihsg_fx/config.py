"""Central configuration for the IHSG vs USD/IDR study.

Changing a single value here updates the CLI, the notebook helpers and tests.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "raw_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Data window ---------------------------------------------------------
# Defaults only: the CLI (--start/--end) and the notebook override these.
START_DATE = "2019-01-01"
END_DATE = "2026-10-01"  # exclusive, matches yfinance semantics
DATA_SOURCE = "csv"      # "csv" (offline, reproducible) or "yfinance" (live)
IHSG_TICKER = "^JKSE"
FX_TICKER = "USDIDR=X"
DATE_COLUMN = "Date"
PRICE_COLUMNS = ["USDIDR", "IHSG"]

# --- Model settings ------------------------------------------------------
MAX_LAG = 5            # trading days of lagged FX returns to include
HAC_MAXLAGS = 5        # Newey-West bandwidth
GRANGER_MAXLAG = 5     # lags tested in the Granger causality test
ALPHA = 0.05           # significance level used throughout
