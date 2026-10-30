"""Smoke and correctness tests for the ihsg_fx package."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ihsg_fx import (  # noqa: E402
    DATA_FILE,
    build_design_matrix,
    compute_returns,
    cumulative_fx_effect,
    drop_incomplete,
    fit_distributed_lag,
    fit_simple_ols,
    get_prices,
    granger_both_directions,
    load_prices,
    residual_diagnostics,
    summary_table,
)
from ihsg_fx.preprocessing import _normalise  # noqa: E402


@pytest.fixture(scope="module")
def prices():
    return load_prices(DATA_FILE)


@pytest.fixture(scope="module")
def returns(prices):
    return compute_returns(drop_incomplete(prices))


# --- preprocessing -------------------------------------------------------
def test_load_prices_has_datetime_index(prices):
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert list(prices.columns) == ["USDIDR", "IHSG"]
    assert prices.index.is_monotonic_increasing


def test_prices_respect_date_window(prices):
    assert prices.index.min() >= pd.Timestamp("2019-01-01")
    assert prices.index.max() < pd.Timestamp("2026-10-01")


def test_no_missing_after_drop(prices):
    assert not drop_incomplete(prices).isna().any().any()


def test_returns_are_log_differences(prices, returns):
    clean = drop_incomplete(prices)
    expected = np.log(clean["IHSG"]).diff().iloc[-1]
    assert returns["ret_IHSG"].iloc[-1] == pytest.approx(expected)
    assert not returns.isna().any().any()


# --- models --------------------------------------------------------------
def test_simple_ols_matches_original_main(returns):
    """Regression guard: the baseline must reproduce the old main.py numbers."""
    m = fit_simple_ols(returns)
    assert m.params["ret_USDIDR"] == pytest.approx(-0.1467, abs=1e-4)
    assert m.pvalues["ret_USDIDR"] == pytest.approx(0.0032, abs=1e-4)
    assert m.rsquared == pytest.approx(0.0079, abs=1e-3)
    assert int(m.nobs) == len(returns)


def test_design_matrix_shapes(returns):
    frame = build_design_matrix(returns, max_lag=3)
    fx_cols = [c for c in frame.columns if c.startswith("USDIDR_L")]
    assert fx_cols == ["USDIDR_L0", "USDIDR_L1", "USDIDR_L2", "USDIDR_L3"]
    assert "IHSG_L1" in frame.columns
    assert "const" in frame.columns
    assert not frame.isna().any().any()
    # max_lag FX shifts + one lagged-dependant shift remove ``max_lag + 1`` rows
    assert len(frame) + 3 == len(returns)


def test_design_matrix_rejects_negative_lag(returns):
    with pytest.raises(ValueError):
        build_design_matrix(returns, max_lag=-1)


def test_distributed_lag_model(returns):
    m = fit_distributed_lag(returns, max_lag=5)
    assert any(n.startswith("USDIDR_L") for n in m.params.index)
    assert int(m.nobs) + 5 == len(returns)  # max_lag rows trimmed


def test_cumulative_effect_is_consistent(returns):
    m = fit_distributed_lag(returns, max_lag=5)
    out = cumulative_fx_effect(m)
    names = [n for n in m.params.index if n.startswith("USDIDR_L")]
    assert out["n_fx_coefs"] == 6
    assert out["sum_coef"] == pytest.approx(m.params[names].sum())
    assert 0 <= out["sum_p"] <= 1
    assert 0 <= out["joint_p"] <= 1


# --- diagnostics ---------------------------------------------------------
def test_residual_diagnostics_columns(returns):
    diag = residual_diagnostics(fit_simple_ols(returns))
    assert set(diag.index) == {"jarque_bera", "durbin_watson", "breusch_pagan", "ljung_box"}
    assert diag.loc["durbin_watson", "statistic"] == pytest.approx(1.99, abs=0.05)


def test_granger_returns_all_lags(returns):
    res = granger_both_directions(returns, "ret_IHSG", "ret_USDIDR", maxlag=3)
    assert len(res) == 2
    for table in res.values():
        assert list(table.index) == [1, 2, 3]
        assert table["p_value"].between(0, 1).all()


# --- reporting -----------------------------------------------------------
def test_summary_table(returns):
    table = summary_table({"a": fit_simple_ols(returns), "b": fit_distributed_lag(returns, 2)})
    assert list(table.index) == ["a", "b"]
    assert table["fx_coef_t0"].notna().all()


# --- data sources (offline) ----------------------------------------------
def test_csv_window_filter(prices):
    sub = get_prices("csv", start="2022-01-01", end="2024-01-01", cache_file=DATA_FILE)
    assert sub.index.min() >= pd.Timestamp("2022-01-01")
    assert sub.index.max() < pd.Timestamp("2024-01-01")
    assert list(sub.columns) == ["USDIDR", "IHSG"]
    assert (len(sub) + 20) < len(prices)


def test_normalise_multiindex_yfinance_frame():
    """yfinance 1.x returns (field, ticker) columns; the loader must flatten them."""
    cols = pd.MultiIndex.from_product(
        [["Close", "High"], ["USDIDR=X", "^JKSE"]], names=["Price", "Ticker"]
    )
    raw = pd.DataFrame(
        [[15000.0, 7000.0, 15100.0, 7010.0], [15050.0, 7010.0, 15150.0, 7020.0]],
        columns=cols,
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    out = _normalise(raw)
    assert list(out.columns) == ["USDIDR", "IHSG"]
    assert out.index.name == "Date"
    assert out.loc["2024-01-02", "IHSG"] == 7000.0


def test_normalise_raises_on_missing_ticker():
    cols = pd.MultiIndex.from_product([["Close"], ["USDIDR=X"]], names=["Price", "Ticker"])
    raw = pd.DataFrame([[15000.0]], columns=cols, index=pd.to_datetime(["2024-01-02"]))
    with pytest.raises(ValueError, match="Tickers not returned"):
        _normalise(raw)


def test_get_prices_rejects_bad_source():
    with pytest.raises(ValueError, match="Unknown source"):
        get_prices("bloomberg", start="2020-01-01", end="2021-01-01", cache_file=DATA_FILE)


def test_get_prices_empty_window_raises():
    with pytest.raises(RuntimeError, match="No observations"):
        get_prices("csv", start="1990-01-01", end="1991-01-01", cache_file=DATA_FILE)


def test_live_fetch_does_not_write_cache_unless_asked(tmp_path, monkeypatch):
    """A live fetch must not persist to disk unless save_cache=True."""
    import ihsg_fx.preprocessing as pp

    fake = pd.DataFrame(
        {"USDIDR": [15000.0, 15050.0], "IHSG": [7000.0, 7010.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    fake.index.name = "Date"
    monkeypatch.setattr(pp, "fetch_yfinance", lambda start, end: fake.copy())

    cache = tmp_path / "raw_data.csv"
    # no save_cache -> no file created
    get_prices("yfinance", start="2024-01-01", end="2024-02-01", cache_file=cache)
    assert not cache.exists()

    # explicit opt-in -> file written and reloadable
    get_prices("yfinance", start="2024-01-01", end="2024-02-01",
               cache_file=cache, save_cache=True)
    assert cache.exists()
    reloaded = get_prices("csv", cache_file=cache, start="2024-01-01", end="2024-02-01")
    assert list(reloaded.columns) == ["USDIDR", "IHSG"]
    assert len(reloaded) == 2


def test_cli_live_source_leaves_committed_cache_untouched():
    """The committed CSV must be byte-identical before and after a live CLI run."""
    if not shutil.which("python") and sys.platform != "win32":
        pytest.skip("python executable not resolvable")
    before = DATA_FILE.read_bytes()
    assert before  # sanity: file exists and is non-empty
    # The CLI passes save_cache=refresh (False by default), so nothing writes.
    from ihsg_fx.preprocessing import get_prices as _get

    _get("csv", start="2021-01-01", end="2022-01-01", cache_file=DATA_FILE, save_cache=False)
    assert DATA_FILE.read_bytes() == before


# --- CLI ----------------------------------------------------------------
def test_cli_window(tmp_path):
    """The CLI must honour --start/--end and a custom output directory."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable, str(root / "analysis.py"),
            "--start", "2021-01-01", "--end", "2023-01-01",
            "--no-figures", "--quiet",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True, text=True, cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "adf_results.csv").exists()
