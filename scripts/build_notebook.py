"""Generate notebooks/ihsg_fx_analysis.ipynb (keeps the .ipynb reproducible from code)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ihsg_fx_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md("""
# IHSG vs USD/IDR — Relationship Analysis

Analysis of the daily relationship between the Jakarta Composite Index (IHSG) and the
USD/IDR exchange rate, 2019-01-01 → 2026-09-30.

**Research questions**

1. Is there a contemporaneous relationship between daily FX moves and IHSG returns?
2. Do FX returns affect IHSG with a delay (distributed lag)?
3. What is the *cumulative* effect of an FX move over the following week?
4. Which direction does causality run (FX → IHSG, IHSG → FX, or both)?

**Layout** — the logic lives in `src/ihsg_fx/`; this notebook only calls it and interprets it.
For a non-interactive run: `python analysis.py`.
"""),
    md("## 0. Setup"),
    code("""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from ihsg_fx import (  # noqa: E402
    DATA_FILE, MAX_LAG, GRANGER_MAXLAG,
    adf_table, compute_returns, drop_incomplete, get_prices,
    fit_simple_ols, fit_distributed_lag, cumulative_fx_effect, summary_table,
    residual_diagnostics, granger_both_directions,
)
from ihsg_fx import plots  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
print("data file:", DATA_FILE)
"""),
    md("""
## 1. Choose the data window and source

Change these three values to run the whole model on a different period.

- `SOURCE = "csv"` — offline, reproducible (uses `data/raw_data.csv`).
- `SOURCE = "yfinance"` — live download for exactly this window.
- `SAVE_TO_CACHE = True` — also overwrite `data/raw_data.csv` (only for `yfinance`).

> The window is **`[START, END)`** — `END` is exclusive, matching yfinance.
"""),
    code("""
START = "2019-01-01"
END = "2026-10-01"     # exclusive
SOURCE = "csv"         # "csv" or "yfinance"
SAVE_TO_CACHE = False
"""),
    code("""
prices_raw = get_prices(
    source=SOURCE,
    start=START,
    end=END,
    cache_file=DATA_FILE,
    save_cache=SAVE_TO_CACHE,
)
prices = drop_incomplete(prices_raw)
ret = compute_returns(prices)

print(f"source          : {SOURCE}")
print(f"window          : {START} -> {END} (end exclusive)")
print(f"raw rows        : {len(prices_raw)}")
print(f"complete rows   : {len(prices)}")
print(f"return obs      : {len(ret)}")
prices_raw.isna().sum().to_frame("missing")
"""),
    md("""
## 2. Data notes

`data/raw_data.csv` holds daily closes from Yahoo Finance. Missing values occur when one
market was open and the other was closed, so we keep only days where **both** series have
an observation — the intersecting trading calendar. No interpolation is used.
"""),
    md("### Prices"),
    code("""
prices.describe().T
"""),
    md("### Returns"),
    code("""
ret.describe().T
"""),
    md("## 2. Exploratory analysis"),
    code("""
plots.plot_prices(prices);
plt.show()
"""),
    code("""
plots.plot_return_series(ret);
plt.show()
"""),
    code("""
plots.plot_return_scatter(ret);
plt.show()
"""),
    code("""
plots.plot_return_distributions(ret);
plt.show()
"""),
    md("""
**Reading the charts.** The levels plot shows both series trending, so any correlation on
levels is spurious — hence the analysis uses log returns. Returns are centred on zero and
strongly leptokurtic (fat tails), which is why the scatter shows only a faint negative slope.
"""),
    md("""
## 3. Stationarity (ADF)

H0 = unit root (non-stationary). Rejecting H0 means the series is stationary.
Levels are expected to be non-stationary; returns should be stationary.
"""),
    code("""
adf = adf_table({
    "IHSG (level)": prices["IHSG"],
    "USDIDR (level)": prices["USDIDR"],
    "ret_IHSG": ret["ret_IHSG"],
    "ret_USDIDR": ret["ret_USDIDR"],
})
adf
"""),
    md("""
## 4. Baseline model — contemporaneous only

$$r^{IHSG}_t = \\alpha + \\beta \\, r^{USDIDR}_t + \\varepsilon_t$$

Estimated by OLS with **Newey-West (HAC) standard errors** (5 lags) so inference stays
valid despite autocorrelation and heteroskedasticity.
"""),
    code("""
baseline = fit_simple_ols(ret)
print(baseline.summary())
"""),
    md("""
## 5. Distributed-lag model

$$r^{IHSG}_t = \\alpha + \\sum_{k=0}^{K} \\beta_k \\, r^{USDIDR}_{t-k} + \\gamma \\, r^{IHSG}_{t-1} + \\varepsilon_t$$

Why the lags:

- FX trades nearly 24h while IDX closes around 16:00 WIB, so "daily close" pairings are
  partly misaligned.
- Information from late/overnight FX moves reaches the index on the *next* close.
- The lagged IHSG term controls for short-term momentum so the FX coefficients are not
  contaminated by own-series dynamics, and makes the reverse Granger test symmetric.
- HAC fixes *inference*, not *specification*: it protects p-values but leaves the
  point estimates unchanged if lags are omitted. That is the real reason to include them.
"""),
    code("""
dlag = fit_distributed_lag(ret, max_lag=MAX_LAG)
print(dlag.summary())
"""),
    code("""
coefs = pd.DataFrame({
    "coef": dlag.params,
    "std_err": dlag.bse,
    "p_value": dlag.pvalues,
})
coefs.loc[[i for i in coefs.index if i.startswith("USDIDR_L")]]
"""),
    md("""
## 6. Cumulative effect

Individual lag coefficients are noisy and individually insignificant. What matters is the
**sum** — the total response of IHSG over the week to a 1% FX move — and the **joint**
F-test that all FX coefficients are zero together.
"""),
    code("""
eff = cumulative_fx_effect(dlag)
pd.Series({
    "sum of FX coefficients": eff["sum_coef"],
    "SE of sum": eff["sum_se"],
    "t (sum = 0)": eff["sum_t"],
    "p (sum = 0)": eff["sum_p"],
    "joint F": eff["joint_F"],
    "joint p": eff["joint_p"],
})
"""),
    code("""
sum_coef = eff["sum_coef"]
print(f"A 1% depreciation of IDR is associated with a cumulative "
      f"{sum_coef * 0.01 * 100:+.4f}% change in IHSG over {MAX_LAG + 1} trading days.")
"""),
    md("## 7. Residual diagnostics"),
    code("""
diag = pd.concat(
    [residual_diagnostics(baseline).add_prefix("baseline_"),
     residual_diagnostics(dlag).add_prefix("dlag_")],
    axis=1,
)
diag
"""),
    md("""
Normality is rejected almost always for daily returns (fat tails, outliers) — that is
expected and not fatal given ~1.9k observations and HAC errors. Autocorrelation and
heteroskedasticity matter more; the HAC covariance covers both.
"""),
    md("## 8. Model comparison"),
    code("""
summary_table({"baseline": baseline, "distributed_lag": dlag})
"""),
    md("## 9. Granger causality (both directions)"),
    code("""
granger = granger_both_directions(ret, "ret_IHSG", "ret_USDIDR", maxlag=GRANGER_MAXLAG)
for direction, table in granger.items():
    print(f"\\n{direction}")
    print(table)
"""),
    md("""
This is the honest answer to "does FX drive IHSG or the reverse". A significant result in
one direction means past values of that series improve the forecast of the other beyond its
own lags — it is predictive (Granger) causality, not structural causality.
"""),
    md("## 10. Findings"),
    code("""
baseline_b = baseline.params["ret_USDIDR"]
baseline_p = baseline.pvalues["ret_USDIDR"]
print(f"1. Contemporaneous beta = {baseline_b:+.4f} (p = {baseline_p:.4f}), R2 = {baseline.rsquared:.4f}")
print(f"2. Sum of FX lags      = {eff['sum_coef']:+.4f} (p = {eff['sum_p']:.4f})")
print(f"3. Joint FX F-test     = {eff['joint_F']:.3f} (p = {eff['joint_p']:.4f})")
for direction, table in granger.items():
    sig = table[table['significant']]
    print(f"4. Granger {direction}: significant at lags {list(sig.index) if len(sig) else 'none'}")
"""),
    md("""
### Interpretation template

- **Sign convention.** USD/IDR up = rupiah weaker. A negative coefficient means a weaker
  rupiah is associated with lower IHSG returns.
- **Contemporaneous effect** is small but statistically robust; daily R² around 1% is
  normal — the FX rate is one of many drivers of the index.
- **Lag terms** are individually fragile but jointly informative; the joint F-test and the
  sum are the numbers to report, not five separate p-values.
- **Causality** can run both ways: foreign outflows hit the currency and the equity market
  simultaneously. Report Granger results as predictive evidence with a caveat.
- **Diagnostics** rejecting normality/Breusch-Pagan is expected for financial returns; the
  HAC covariance is what keeps inference credible.
"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14.8"},
}
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK)
print(f"wrote {NOTEBOOK} ({len(cells)} cells)")
