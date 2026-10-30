"""Data sources: live Yahoo Finance fetch or the cached CSV.

Every other module only depends on :func:`get_prices`, so switching source or
changing the analysis window never touches models, diagnostics or plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, FX_TICKER, IHSG_TICKER, PRICE_COLUMNS

# yfinance ticker -> column name used throughout the project
TICKER_MAP = {IHSG_TICKER: "IHSG", FX_TICKER: "USDIDR"}


def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance frame into ``Date``-indexed ``USDIDR`` / ``IHSG`` columns.

    Handles both the single- and multi-ticker column layouts that different
    yfinance versions return.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        # Newer yfinance: (field, ticker) or (ticker, field)
        level0 = raw.columns.get_level_values(0)
        if "Close" in level0:
            close = raw["Close"]
        else:
            close = raw.xs("Close", axis=1, level=1)
    else:
        close = raw[["Close"]].copy()

    if isinstance(close, pd.Series):
        close = close.to_frame()

    # Map tickers to our names, then keep only the requested ones.
    close = close.rename(columns=TICKER_MAP)
    missing = [name for name in PRICE_COLUMNS if name not in close.columns]
    if missing:
        raise ValueError(f"Tickers not returned by the data source: {missing}")

    out = close[PRICE_COLUMNS].astype(float)
    out.index = pd.to_datetime(out.index)
    out.index.name = DATE_COLUMN
    return out.sort_index()


def fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """Download daily closes from Yahoo Finance for ``[start, end)``."""
    import yfinance as yf

    raw = yf.download(
        list(TICKER_MAP),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned no data for {start} -> {end}. "
            "Check the dates, connectivity, or that the tickers still exist."
        )
    return _normalise(raw)


def load_prices_csv(path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read the cached CSV, optionally restricted to ``[start, end)``."""
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])

    if df[DATE_COLUMN].duplicated().any():
        raise ValueError(f"Duplicate dates found in {path}")

    df = df.set_index(DATE_COLUMN).sort_index()[PRICE_COLUMNS].astype(float)
    if start is not None:
        df = df.loc[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df.loc[df.index < pd.Timestamp(end)]
    return df


def save_prices(prices: pd.DataFrame, path) -> None:
    """Write a price frame back to CSV (used by ``--refresh``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, float_format="%.10g")


def get_prices(
    source: str = "yfinance",
    start: str | None = None,
    end: str | None = None,
    cache_file=None,
    save_cache: bool = False,
) -> pd.DataFrame:
    """Unified entry point for price data.

    Parameters
    ----------
    source : {"yfinance", "csv"}
        ``yfinance`` downloads live data for the requested window; ``csv`` reads
        the cached file (offline, and reproducible for tests).
    start, end : str, optional
        Analysis window, ``[start, end)``. For ``source="csv"`` the window is a
        filter on the cached dates.
    cache_file : Path, optional
        Where to write the data when ``save_cache=True``.
    save_cache : bool
        Persist the downloaded window (typically the full history) to CSV.
    """
    if start is None or end is None:
        from .config import END_DATE, START_DATE

        start = start or START_DATE
        end = end or END_DATE

    if source == "yfinance":
        prices = fetch_yfinance(start, end)
        if save_cache and cache_file is not None:
            save_prices(prices, cache_file)
    elif source == "csv":
        if cache_file is None:
            raise ValueError("cache_file is required when source='csv'")
        prices = load_prices_csv(cache_file, start=start, end=end)
    else:
        raise ValueError(f"Unknown source {source!r}; use 'yfinance' or 'csv'")

    if prices.empty:
        raise RuntimeError(f"No observations in the window {start} -> {end}")

    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected; check the source data")

    return prices


def drop_incomplete(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only days where every series has an observation (intersecting calendar)."""
    return df.dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded (log) returns with study-friendly column names.

    The leading ``NaN`` created by differencing is dropped so later ``shift``
    operations lose only the number of rows they actually consume.
    """
    if len(prices) < 2:
        raise ValueError("Need at least two price observations to compute returns")

    ret = np.log(prices).diff().iloc[1:]
    ret.columns = ["ret_USDIDR", "ret_IHSG"]
    return ret.dropna()
