"""Reproducible, non-interactive run of the full IHSG vs USD/IDR analysis.

Usage
-----
    python analysis.py                          # cached CSV, default window
    python analysis.py --source yfinance        # live data for the default window
    python analysis.py --start 2020-01-01 --end 2023-01-01
    python analysis.py --refresh                # update data/raw_data.csv from Yahoo
    python analysis.py --no-figures             # skip chart generation
    python analysis.py --quiet                  # only write files

``--source yfinance`` analyses the requested window live, without touching the
cached CSV. ``--refresh`` additionally overwrites ``data/raw_data.csv``.

Replaces the old monolithic ``main.py``; the notebook in ``notebooks/``
walks through the same steps interactively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ihsg_fx import (  # noqa: E402
    DATA_FILE,
    DATA_SOURCE,
    END_DATE,
    GRANGER_MAXLAG,
    MAX_LAG,
    OUTPUT_DIR,
    START_DATE,
    adf_table,
    compute_returns,
    fit_distributed_lag,
    fit_simple_ols,
    get_prices,
    granger_both_directions,
    residual_diagnostics,
    summary_table,
)
from ihsg_fx.models import cumulative_fx_effect  # noqa: E402
from ihsg_fx.preprocessing import drop_incomplete, save_prices  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run(
    show_figures: bool = True,
    quiet: bool = False,
    start: str = START_DATE,
    end: str = END_DATE,
    source: str = DATA_SOURCE,
    refresh: bool = False,
    output_dir: Path | None = None,
) -> dict:
    pd.set_option("display.float_format", lambda v: f"{v:,.6f}")
    output_dir = Path(output_dir or OUTPUT_DIR)

    # --- 1. Data ---------------------------------------------------------
    prices_raw = get_prices(
        source=source,
        start=start,
        end=end,
        cache_file=DATA_FILE,
        save_cache=refresh,
    )
    if refresh:
        save_prices(prices_raw, DATA_FILE)

    missing = prices_raw.isna().sum()
    prices = drop_incomplete(prices_raw)
    ret = compute_returns(prices)

    if not quiet:
        section("1. DATA")
        print(f"Source: {source}   window: {start} -> {end} (end exclusive)")
        print(f"Rows: {len(prices_raw)}   missing per column:\n{missing.to_string()}")
        print(f"Complete rows (intersecting calendar): {len(prices)}")
        print(f"Return observations: {len(ret)}")
        if refresh:
            print(f"Cache refreshed: {DATA_FILE}")
        print(prices.describe().T)
        print(ret.describe().T)

    # --- 2. Stationarity -------------------------------------------------
    adf = adf_table(
        {
            "IHSG (level)": prices["IHSG"],
            "USDIDR (level)": prices["USDIDR"],
            "ret_IHSG": ret["ret_IHSG"],
            "ret_USDIDR": ret["ret_USDIDR"],
        }
    )
    if not quiet:
        section("2. STATIONARITY (ADF, H0 = unit root)")
        print(adf.to_string())

    # --- 3. Baseline model ----------------------------------------------
    baseline = fit_simple_ols(ret)
    if not quiet:
        section("3. BASELINE OLS: ret_IHSG ~ ret_USDIDR (HAC SE)")
        print(baseline.summary())

    # --- 4. Distributed lag ---------------------------------------------
    dlag = fit_distributed_lag(ret, max_lag=MAX_LAG)
    cumulative = cumulative_fx_effect(dlag)
    if not quiet:
        section(f"4. DISTRIBUTED LAG: FX lags 0..{MAX_LAG} + ret_IHSG_t-1 (HAC SE)")
        print(dlag.summary())

    # --- 5. Diagnostics --------------------------------------------------
    diag_baseline = residual_diagnostics(baseline)
    diag_dlag = residual_diagnostics(dlag)
    if not quiet:
        section("5. RESIDUAL DIAGNOSTICS")
        print("Baseline model\n" + diag_baseline.to_string())
        print("\nDistributed-lag model\n" + diag_dlag.to_string())

    # --- 6. Cumulative effect -------------------------------------------
    if not quiet:
        section("6. CUMULATIVE FX EFFECT")
        sig = "significant" if cumulative["sum_p"] < 0.05 else "not significant"
        print(f"Sum of FX coefficients   : {cumulative['sum_coef']:+.4f} "
              f"(SE {cumulative['sum_se']:.4f})")
        print(f"Test sum = 0             : t = {cumulative['sum_t']:.3f}, "
              f"p = {cumulative['sum_p']:.4f}  -> {sig}")
        print(f"Joint F-test all FX = 0  : F = {cumulative['joint_F']:.3f}, "
              f"p = {cumulative['joint_p']:.4f}")

    # --- 7. Model comparison --------------------------------------------
    comparison = summary_table({"baseline": baseline, "distributed_lag": dlag})
    if not quiet:
        section("7. MODEL COMPARISON")
        print(comparison.to_string())

    # --- 8. Granger causality -------------------------------------------
    granger = granger_both_directions(ret, "ret_IHSG", "ret_USDIDR", maxlag=GRANGER_MAXLAG)
    if not quiet:
        section(f"8. GRANGER CAUSALITY (maxlag={GRANGER_MAXLAG})")
        for direction, table in granger.items():
            print(f"\n{direction}\n{table.to_string()}")

    # --- 9. Figures ------------------------------------------------------
    if show_figures:
        from ihsg_fx import plots

        figures = {
            "prices": plots.plot_prices(prices, output_dir),
            "return_scatter": plots.plot_return_scatter(ret, output_dir),
            "return_distributions": plots.plot_return_distributions(ret, output_dir),
            "return_series": plots.plot_return_series(ret, output_dir),
        }
        if not quiet:
            section("9. FIGURES")
            for name in figures:
                print(f"saved {name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    adf.to_csv(output_dir / "adf_results.csv")
    diag_baseline.to_csv(output_dir / "diagnostics_baseline.csv")
    diag_dlag.to_csv(output_dir / "diagnostics_distributed_lag.csv")
    comparison.to_csv(output_dir / "model_comparison.csv")
    for direction, table in granger.items():
        safe = direction.replace(" -> ", "_to_").replace(" ", "_")
        table.to_csv(output_dir / f"granger_{safe}.csv")

    return {
        "window": {"start": start, "end": end, "source": source},
        "prices": prices,
        "returns": ret,
        "adf": adf,
        "baseline": baseline,
        "distributed_lag": dlag,
        "cumulative": cumulative,
        "diagnostics": diag_baseline,
        "comparison": comparison,
        "granger": granger,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IHSG vs USD/IDR analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", default=START_DATE, help=f"window start (default {START_DATE})")
    parser.add_argument("--end", default=END_DATE, help=f"window end, exclusive (default {END_DATE})")
    parser.add_argument(
        "--source",
        choices=["csv", "yfinance"],
        default=DATA_SOURCE,
        help="csv = cached file (offline), yfinance = live download for the window",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="overwrite data/raw_data.csv with freshly downloaded data",
    )
    parser.add_argument("--output-dir", default=None, help="where to write tables/figures")
    parser.add_argument("--no-figures", action="store_true", help="skip chart generation")
    parser.add_argument("--quiet", action="store_true", help="write results without printing")
    args = parser.parse_args()
    run(
        show_figures=not args.no_figures,
        quiet=args.quiet,
        start=args.start,
        end=args.end,
        source=args.source,
        refresh=args.refresh,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
