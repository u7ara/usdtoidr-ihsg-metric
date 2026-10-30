"""Stationarity, residual diagnostics and causality tests."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats

from .config import ALPHA


def adf_test(series: pd.Series, name: str | None = None) -> dict:
    """Augmented Dickey-Fuller test. Rejecting H0 (p < alpha) => series is stationary."""
    result = adfuller(series.dropna(), autolag="AIC", result_object=False)
    return {
        "series": name or series.name,
        "adf_stat": float(result[0]),
        "p_value": float(result[1]),
        "used_lags": int(result[2]),
        "nobs": int(result[3]),
        "stationary": bool(result[1] < ALPHA),
    }


def adf_table(series: dict[str, pd.Series]) -> pd.DataFrame:
    """ADF results for several series, stacked into one table."""
    return pd.DataFrame([adf_test(s, n) for n, s in series.items()]).set_index("series")


def residual_diagnostics(model) -> pd.DataFrame:
    """Normality, autocorrelation and heteroskedasticity checks on OLS residuals."""
    resid = pd.Series(model.resid, index=model.model.data.row_labels)
    exog = model.model.exog

    jb_stat, jb_p = stats.jarque_bera(resid)
    bp_stat, bp_p = het_breuschpagan(resid, exog)[:2]
    dw = durbin_watson(resid)
    lb = acorr_ljungbox(resid, lags=[5], return_df=True)

    return pd.DataFrame(
        {
            "statistic": [jb_stat, dw, bp_stat, lb["lb_stat"].iloc[0]],
            "p_value": [jb_p, np.nan, bp_p, lb["lb_pvalue"].iloc[0]],
            "passed": [jb_p > ALPHA, 1.5 < dw < 2.5, bp_p > ALPHA, lb["lb_pvalue"].iloc[0] > ALPHA],
            "note": [
                "Jarque-Bera: normal residuals",
                "Durbin-Watson: no serial correlation (ideal ~2)",
                "Breusch-Pagan: homoskedastic residuals",
                "Ljung-Box(5): no residual autocorrelation",
            ],
        },
        index=["jarque_bera", "durbin_watson", "breusch_pagan", "ljung_box"],
    )


def granger_causality(df: pd.DataFrame, cause: str, effect: str, maxlag: int = 5) -> pd.DataFrame:
    """Granger test: does ``cause`` help predict ``effect`` beyond ``effect``'s own lags?

    ``df`` must contain both columns in date order. H0: no Granger causality.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = grangercausalitytests(df[[effect, cause]].dropna(), maxlag=maxlag)

    rows = []
    for lag, (tests, _) in res.items():
        f_stat, p_value, _, _ = tests["ssr_ftest"]
        rows.append({"lag": lag, "F": f_stat, "p_value": p_value, "significant": p_value < ALPHA})
    return pd.DataFrame(rows).set_index("lag")


def granger_both_directions(df: pd.DataFrame, col_a: str, col_b: str, maxlag: int = 5) -> dict:
    """Run the Granger test in both directions and label each result set."""
    return {
        f"{col_b} -> {col_a}": granger_causality(df, cause=col_b, effect=col_a, maxlag=maxlag),
        f"{col_a} -> {col_b}": granger_causality(df, cause=col_a, effect=col_b, maxlag=maxlag),
    }
