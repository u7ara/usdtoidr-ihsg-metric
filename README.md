# IHSG vs USD/IDR

Econometric analysis of the daily relationship between the Jakarta Composite Index
(`^JKSE`) and the USD/IDR exchange rate, 2019-01-01 → 2026-09-30.

## Research questions

1. Is there a **contemporaneous** relationship between daily FX moves and IHSG returns?
2. Do FX returns affect IHSG with a **delay** (distributed lag)?
3. What is the **cumulative** effect of an FX move over the following week?
4. Which direction does **causality** run (FX → IHSG, IHSG → FX, or both)?

## Project layout

```
IHSG/
├── analysis.py                 # one-shot CLI: runs everything, prints report, saves outputs/
├── data/
│   └── raw_data.csv            # daily closes from Yahoo Finance (committed)
├── notebooks/
│   └── ihsg_fx_analysis.ipynb  # interactive walkthrough + interpretation
├── scripts/
│   └── build_notebook.py       # regenerates the notebook from code
├── src/
│   └── ihsg_fx/
│       ├── config.py           # dates, paths, lag settings — single source of truth
│       ├── preprocessing.py    # sources (csv/yfinance), cleaning, log returns
│       ├── models.py           # OLS, distributed lag, cumulative effect, comparison
│       ├── diagnostics.py      # ADF, Jarque-Bera, DW, Breusch-Pagan, Ljung-Box, Granger
│       └── plots.py            # EDA charts
├── outputs/                    # generated tables (.csv) and figures (.png)
├── tests/
│   └── test_pipeline.py        # pytest suite (includes a regression guard on old results)
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# default: cached CSV, full window
python analysis.py

# any window you like (end is EXCLUSIVE)
python analysis.py --start 2020-01-01 --end 2023-01-01

# live data straight from Yahoo Finance for that window (no cache write)
python analysis.py --source yfinance --start 2023-01-01 --end 2025-01-01

# refresh the cached CSV in data/raw_data.csv
python analysis.py --refresh

# keep a separate set of outputs per window
python analysis.py --start 2020-01-01 --end 2023-01-01 --output-dir outputs/2020_2023

# other options
python analysis.py --no-figures
python analysis.py --quiet

# tests
python -m pytest tests -q

# interactive exploration
jupyter lab notebooks/ihsg_fx_analysis.ipynb
```

`--source yfinance` fetches only the requested window and **never** writes to disk unless
you pass `--refresh`. That keeps the committed CSV as a fixed benchmark while letting you
run any period live.

### Changing the window in Python / the notebook

```python
from ihsg_fx import get_prices, drop_incomplete, compute_returns

prices = get_prices("yfinance", start="2020-01-01", end="2023-01-01")
ret = compute_returns(drop_incomplete(prices))
```

In the notebook, edit the `START`, `END`, `SOURCE` cells at the top; everything
downstream follows automatically.

## Method summary

| Step | Test / model | Why |
|---|---|---|
| Stationarity | ADF on levels and returns | avoid spurious regression on trending levels |
| Baseline | `ret_IHSG ~ ret_USDIDR` OLS, **HAC** (Newey-West, 5 lags) | contemporaneous effect, robust inference |
| Distributed lag | `ret_IHSG ~ Σₖ ret_USDIDR_{t-k} (k=0..5) + ret_IHSG_{t-1}` | FX and IDX close at different hours; information arrives with a delay |
| Cumulative effect | sum of FX coefficients + joint F-test | total weekly response, not five noisy p-values |
| Diagnostics | Jarque-Bera, Durbin-Watson, Breusch-Pagan, Ljung-Box | check model fit |
| Causality | Granger test, both directions, lags 1–5 | FX → IHSG is not the only possible direction |

### Why lags at all?

- USD/IDR trades nearly 24h; IDX closes ~16:00 WIB, so "same-day" closes are misaligned.
- Overnight FX moves may only reach the index on the next close.
- HAC errors fix **inference**, not **specification** — they protect p-values but do not
  correct omitted-variable bias in the point estimates.
- The lag set is fixed a priori (0–5 trading days) and reported in full, avoiding
  data-driven lag selection that would manufacture spurious significance.

## Data sources

Two interchangeable sources, selected with `--source` (or `get_prices(source=...)`):

| Source | Behaviour | Use for |
|---|---|---|
| `csv` (default) | reads `data/raw_data.csv`, filtered to the window | offline, reproducible, tests |
| `yfinance` | live download for the window | any custom period, fresh data |

The loader normalises yfinance's MultiIndex columns (`(Price, Ticker)`) into a flat
`Date`-indexed frame with `USDIDR` / `IHSG`, and raises a clear error if no data or an
unexpected ticker comes back.

## Data notes

- `data/raw_data.csv` contains **2018 rows**, of which 1866 have both prices;
  IHSG is missing 150 times and USD/IDR twice (non-overlapping market calendars).
- Returns are log differences, so there are **1865** return observations.
- Missing values are handled by keeping only the intersecting trading calendar; no
  interpolation is used.

## Key results (as of the last run)

| Metric | Value |
|---|---|
| Contemporaneous β (HAC) | −0.147 (p = 0.003) |
| Baseline R² | 0.008 |
| Sum of FX lags | −0.303 (p = 0.322) |
| Joint FX F-test | F = 2.21 (p = 0.039) |
| Granger FX → IHSG | not significant at lags 1–5 |
| Granger IHSG → FX | significant at lags 1–5 |

**Interpretation.** A weaker rupiah is associated with *lower* IHSG returns on the same
day, and the FX coefficients are jointly informative, but the *cumulative* effect is not
statistically distinguishable from zero. Granger tests show the predictive flow runs
predominantly from IHSG to the exchange rate, not the reverse — consistent with foreign
flows hitting both markets simultaneously rather than FX causing the index. Daily R² near
1% is normal for single-factor models on equity returns.

## Caveats

- Granger causality is **predictive**, not structural.
- Daily returns are non-normal with fat tails; diagnostics reject normality and
  homoskedasticity, which is expected. HAC standard errors keep inference credible.
- Missing values are handled by keeping only the intersecting trading calendar; no
  interpolation is used, so results are not affected by imputation choices.
