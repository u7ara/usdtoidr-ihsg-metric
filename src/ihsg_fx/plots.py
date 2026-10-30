"""Exploratory charts. All figures are saved to ``outputs/`` and returned."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks override this themselves

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import OUTPUT_DIR


def _save(fig, name: str, outdir: Path = OUTPUT_DIR):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_prices(prices: pd.DataFrame, outdir: Path = OUTPUT_DIR):
    """IHSG and USD/IDR levels on twin axes."""
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(prices.index, prices["IHSG"], color="tab:blue", label="IHSG")
    ax1.set_ylabel("IHSG", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(prices.index, prices["USDIDR"], color="tab:red", label="USD/IDR")
    ax2.set_ylabel("USD/IDR", color="tab:red")
    ax1.set_title("IHSG vs USD/IDR (levels)")
    _save(fig, "01_prices.png", outdir)
    return fig


def plot_return_scatter(ret: pd.DataFrame, outdir: Path = OUTPUT_DIR):
    """Scatter of daily returns with a fitted line."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.regplot(
        x="ret_USDIDR",
        y="ret_IHSG",
        data=ret,
        ax=ax,
        scatter_kws={"alpha": 0.3, "s": 12},
        line_kws={"color": "crimson"},
    )
    ax.set_title("Daily return: USD/IDR vs IHSG")
    _save(fig, "02_return_scatter.png", outdir)
    return fig


def plot_return_distributions(ret: pd.DataFrame, outdir: Path = OUTPUT_DIR):
    """Histograms of both return series."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col in zip(axes, ret.columns):
        sns.histplot(ret[col], bins=60, kde=True, ax=ax)
        ax.set_title(col)
    fig.suptitle("Return distributions")
    _save(fig, "03_return_distributions.png", outdir)
    return fig


def plot_return_series(ret: pd.DataFrame, outdir: Path = OUTPUT_DIR):
    """Return time series side by side."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, col in zip(axes, ret.columns):
        ax.plot(ret.index, ret[col], lw=0.6)
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_ylabel(col)
    fig.suptitle("Daily log returns")
    _save(fig, "04_return_series.png", outdir)
    return fig
