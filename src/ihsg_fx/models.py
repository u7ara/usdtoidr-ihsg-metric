"""Model specifications: contemporaneous OLS and distributed-lag OLS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import HAC_MAXLAGS, MAX_LAG


def _hac_fit(y: pd.Series, X: pd.DataFrame, maxlags: int = HAC_MAXLAGS):
    """OLS with Newey-West (HAC) covariance robust to autocorrelation and heteroskedasticity."""
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


def fit_simple_ols(ret: pd.DataFrame, maxlags: int = HAC_MAXLAGS):
    """Baseline: IHSG return on the *same-day* FX return only.

    ``ret_IHSG_t = a + b * ret_USDIDR_t + e_t``
    """
    y = ret["ret_IHSG"]
    X = sm.add_constant(ret[["ret_USDIDR"]])
    return _hac_fit(y, X, maxlags=maxlags)


def build_design_matrix(ret: pd.DataFrame, max_lag: int = MAX_LAG) -> pd.DataFrame:
    """Build contemporaneous + lagged FX returns plus a lagged IHSG term.

    ``ret_IHSG_t = a + sum_k b_k * ret_USDIDR_{t-k} + g * ret_IHSG_{t-1} + e_t``

    The lagged dependent variable is included on purpose: the fully lagged
    regressor set makes it cheap to also test the reverse (Granger) direction.
    It is reported separately so its effect is visible.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")

    X = pd.DataFrame(index=ret.index)
    for lag in range(max_lag + 1):
        X[f"USDIDR_L{lag}"] = ret["ret_USDIDR"].shift(lag)
    X["IHSG_L1"] = ret["ret_IHSG"].shift(1)
    y = ret["ret_IHSG"].rename("ret_IHSG")

    frame = sm.add_constant(X.assign(ret_IHSG=y), has_constant="add").dropna()
    return frame


def fit_distributed_lag(ret: pd.DataFrame, max_lag: int = MAX_LAG, maxlags: int = HAC_MAXLAGS):
    """Distributed-lag regression with HAC errors.

    Returns a fitted statsmodels result whose ``fittedvalues`` alias is the IHSG return.
    """
    frame = build_design_matrix(ret, max_lag=max_lag)
    y = frame["ret_IHSG"]
    X = frame.drop(columns="ret_IHSG")
    model = _hac_fit(y, X, maxlags=maxlags)
    model.extra = {"max_lag": max_lag, "nobs": int(model.nobs)}
    return model


def cumulative_fx_effect(model) -> dict:
    """Summarise the FX coefficients of a distributed-lag model.

    Returns
    -------
    dict with
        ``sum_coef``  – sum of contemporaneous + lagged FX coefficients
                        (cumulative effect of a 1-unit FX move),
        ``sum_se/t/p`` – test of ``sum(coef) = 0`` using the HAC covariance,
        ``joint_F/joint_p`` – joint F-test that *all* FX coefficients are zero,
        ``n_fx_coefs`` – number of FX terms tested.
    """
    names = [n for n in model.params.index if n.startswith("USDIDR_L")]
    names.sort(key=lambda n: int(n.split("L")[-1]))
    if not names:
        raise ValueError("No USDIDR_L* coefficients found in the model")

    f_test = model.f_test(", ".join(names) + " = 0")

    # The sum is a linear combination, so its SE comes from the HAC covariance matrix.
    cov = np.asarray(model.cov_params().loc[names, names], dtype=float)
    ones = np.ones(len(names))
    total = float(model.params[names].sum())
    se = float(np.sqrt(ones @ cov @ ones))
    t_stat = total / se if se > 0 else float("nan")
    p_value = 2 * stats.norm.sf(abs(t_stat)) if se > 0 else float("nan")

    return {
        "sum_coef": total,
        "sum_se": se,
        "sum_t": t_stat,
        "sum_p": float(p_value),
        "joint_F": float(np.squeeze(f_test.fvalue)),
        "joint_p": float(np.squeeze(f_test.pvalue)),
        "n_fx_coefs": len(names),
    }


def summary_table(models: dict) -> pd.DataFrame:
    """Side-by-side comparison of several fitted models."""
    rows = []
    for name, m in models.items():
        row = {
            "model": name,
            "nobs": int(m.nobs),
            "r2": float(m.rsquared),
            "adj_r2": float(m.rsquared_adj),
            "aic": float(m.aic),
            "bic": float(m.bic),
            "fx_coef_t0": float(m.params.get("USDIDR_L0", m.params.get("ret_USDIDR", np.nan))),
            "fx_p_t0": float(m.pvalues.get("USDIDR_L0", m.pvalues.get("ret_USDIDR", np.nan))),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
