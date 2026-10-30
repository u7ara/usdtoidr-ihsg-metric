"""IHSG vs USD/IDR econometric analysis package."""

from .config import (
    ALPHA,
    DATA_FILE,
    DATA_SOURCE,
    END_DATE,
    FX_TICKER,
    GRANGER_MAXLAG,
    IHSG_TICKER,
    MAX_LAG,
    OUTPUT_DIR,
    START_DATE,
)
from .diagnostics import (
    adf_table,
    adf_test,
    granger_both_directions,
    granger_causality,
    residual_diagnostics,
)
from .models import (
    build_design_matrix,
    cumulative_fx_effect,
    fit_distributed_lag,
    fit_simple_ols,
    summary_table,
)
from .preprocessing import (
    compute_returns,
    drop_incomplete,
    fetch_yfinance,
    get_prices,
    load_prices_csv,
    save_prices,
)

__all__ = [
    "ALPHA",
    "DATA_FILE",
    "DATA_SOURCE",
    "END_DATE",
    "FX_TICKER",
    "GRANGER_MAXLAG",
    "IHSG_TICKER",
    "MAX_LAG",
    "OUTPUT_DIR",
    "START_DATE",
    "adf_table",
    "adf_test",
    "build_design_matrix",
    "compute_returns",
    "cumulative_fx_effect",
    "download_prices",
    "drop_incomplete",
    "fetch_yfinance",
    "fit_distributed_lag",
    "fit_simple_ols",
    "get_prices",
    "granger_both_directions",
    "granger_causality",
    "load_prices",
    "load_prices_csv",
    "residual_diagnostics",
    "save_prices",
    "summary_table",
]


def load_prices(path, start: str = START_DATE, end: str = END_DATE):
    """Backwards-compatible alias for :func:`load_prices_csv`."""
    return load_prices_csv(path, start=start, end=end)


def download_prices(start: str = START_DATE, end: str = END_DATE):
    """Fetch a window from Yahoo Finance and refresh ``data/raw_data.csv``."""
    prices = fetch_yfinance(start, end)
    save_prices(prices, DATA_FILE)
    return prices
