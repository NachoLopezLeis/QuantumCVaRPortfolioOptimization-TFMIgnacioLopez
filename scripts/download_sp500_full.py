#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Script  : download_sp500_full.py  (Data Download - Full S&P 500)
# ============================================================================
#
# Description
# -----------
# Downloads daily adjusted close prices for all S&P 500 constituents,
# computes log returns, and saves to data/returns_sp500_full.csv.
#
# Usage
# -----
#   cd PROJECT_ROOT
#   python scripts/download_sp500_full.py
#   python scripts/download_sp500_full.py --start 2020-01-01 --end 2025-12-31
#   python scripts/download_sp500_full.py --min-obs 400  # drop sparse tickers
#
# Requirements
# ------------
#   pip install yfinance pandas numpy
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("ERROR: yfinance not installed. Run: pip install yfinance")


# ============================================================================
# S&P 500 CONSTITUENTS (as of March 2026 — static list for reproducibility)
# ============================================================================

SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "BRK-B", "JPM", "LLY", "V", "UNH", "MA", "XOM", "JNJ", "COST", "HD",
    "PG", "ABBV", "WMT", "NFLX", "BAC", "KO", "CRM", "MRK", "CVX", "PEP",
    "TMO", "AMD", "ADBE", "ACN", "LIN", "MCD", "CSCO", "ABT", "DHR", "WFC",
    "TXN", "ORCL", "PM", "QCOM", "INTU", "CMCSA", "GE", "VZ", "INTC", "DIS",
    "AMGN", "PFE", "IBM", "AMAT", "NOW", "CAT", "GS", "HON", "UPS", "T",
    "BA", "ISRG", "BKNG", "MS", "BLK", "AXP", "LOW", "NEE", "SPGI", "RTX",
    "MDT", "GILD", "SYK", "BMY", "DE", "CB", "LMT", "SCHW", "ADI", "LRCX",
    "MMC", "PLD", "C", "TJX", "VRTX", "ADP", "MDLZ", "REGN", "SLB", "ETN",
    "KLAC", "BDX", "EOG", "CI", "CME", "BSX", "SO", "DUK", "SNPS", "AON",
    "CDNS", "MO", "CL", "ICE", "SHW", "EQIX", "PYPL", "PNC", "APD", "MCO",
    "FDX", "CMG", "ORLY", "MCK", "NOC", "USB", "MMM", "EMR", "MCHP", "GD",
    "PSA", "MSI", "NSC", "WM", "AZO", "PGR", "HUM", "CTAS", "CCI", "TDG",
    "ITW", "WELL", "APH", "ROST", "OXY", "PCAR", "ROP", "TGT", "NXPI",
    "MNST", "ADSK", "AIG", "SPG", "SRE", "KHC", "MPC", "CARR", "TEL", "TRV",
    "HCA", "ECL", "AFL", "AEP", "FTNT", "PSX", "DHI", "DXCM", "PCG", "NUE",
    "EW", "IDXX", "BK", "STZ", "O", "MSCI", "KMB", "NEM", "CTSH", "COF",
    "AMP", "PRU", "VLO", "PAYX", "GIS", "ALL", "D", "YUM", "SYY", "FAST",
    "FIS", "CMI", "OTIS", "DOW", "A", "CBRE", "HSY", "PEG", "EXC", "KEYS",
    "RSG", "GPN", "DLR", "VRSK", "AMT", "EL", "CPRT", "XEL", "MKTX", "HAL",
    "ED", "PPG", "EA", "ON", "KMI", "DVN", "IQV", "GEHC", "MLM", "RCL",
    "FANG", "WEC", "AWK", "BIIB", "CDW", "ANSS", "GWW", "IR", "ODFL",
    "EXR", "MTD", "FITB", "HPQ", "BKR", "TSCO", "DD", "DAL", "ROK", "ES",
    "AVB", "HIG", "SBAC", "WBD", "STT", "RMD", "WTW", "BAX", "VMC", "GLW",
    "FTV", "EFX", "IFF", "ZBH", "LH", "BR", "VICI", "WAB", "MTB", "DTE",
    "LYB", "PTC", "EIX", "PHM", "RF", "MKC", "HUBB", "TRGP", "CNP", "HPE",
    "CAH", "VLTO", "CLX", "MOH", "MAA", "PPL", "WAT", "STE", "NTAP", "FE",
    "HBAN", "CFG", "K", "WRB", "DGX", "IP", "PKG", "TROW", "HOLX",
    "STLD", "CINF", "KEY", "ATO", "TSN", "TER", "LUV", "J", "BALL",
    "ALGN", "SBNY", "TXT", "ARE", "SWKS", "NTRS", "POOL", "WY", "OMC",
    "INVH", "AKAM", "CE", "ULTA", "EPAM", "JBHT", "GRMN", "DRI", "SNA",
    "TYL", "PEAK", "KIM", "AVY", "EVRG", "NDAQ", "FICO", "BRO", "HST",
    "REG", "CMS", "LNT", "AES", "NI", "JKHY", "BG", "DPZ", "UDR",
    "EMN", "MAS", "TECH", "HRL", "AAL", "UAL", "CCL", "WDC", "SWK",
    "PNR", "CPT", "MGM", "BWA", "APA", "FMC", "IVZ", "HWM", "ALLE",
    "GL", "BF-B", "MHK", "SEE", "ALB", "RHI", "CRL", "CTVA", "NCLH",
    "BBWI", "FOXA", "FOX", "PARA", "VFC", "ZION", "WYNN",
    "NWS", "NWSA", "BEN", "DVA", "DISH", "LKQ", "MTCH", "CZR",
    "XRAY", "ETSY", "GEN", "HII", "TAP", "CPB", "WHR", "FRT",
    "PAYC", "CHRW", "HSIC", "NRG", "L", "AIZ", "GNRC", "ROL",
    "CBOE", "TFX", "AAP", "SEDG",
]


def find_project_root() -> Path:
    """Find project root by looking for config/ directory."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for p in candidates:
        if (p / "config").is_dir():
            return p
    raise FileNotFoundError("Cannot find project root (no config/ directory)")


def download_sp500(
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    min_observations: int = 400,
    max_missing_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Download S&P 500 daily returns.

    Parameters
    ----------
    start_date, end_date : str
        Date range for price download.
    min_observations : int
        Minimum non-NaN return observations required.
    max_missing_pct : float
        Maximum percentage of missing data allowed per ticker.

    Returns
    -------
    returns_df : pd.DataFrame
        Daily log returns (T x N), NaN-cleaned.
    """
    # Remove duplicates from ticker list
    tickers = sorted(set(SP500_TICKERS))
    n_total = len(tickers)

    print(f"Downloading {n_total} S&P 500 tickers: {start_date} to {end_date}")

    # Download in batches to avoid timeouts
    batch_size = 50
    all_prices = []
    for i in range(0, n_total, batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} tickers...")
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(data.columns, pd.MultiIndex):
                prices = data["Close"]
            else:
                prices = data
            all_prices.append(prices)
        except Exception as e:
            print(f"  WARNING: Batch failed: {e}")

    if not all_prices:
        raise RuntimeError("No data downloaded")

    prices_df = pd.concat(all_prices, axis=1)

    # Drop tickers with too many missing values
    missing_pct = prices_df.isna().mean() * 100
    valid_tickers = missing_pct[missing_pct <= max_missing_pct].index.tolist()
    dropped = set(prices_df.columns) - set(valid_tickers)
    if dropped:
        print(f"  Dropped {len(dropped)} tickers (>{max_missing_pct}% missing): "
              f"{sorted(dropped)[:10]}...")
    prices_df = prices_df[valid_tickers]

    # Forward-fill, then back-fill remaining NaNs
    prices_df = prices_df.ffill().bfill()

    # Compute log returns
    returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

    # Drop tickers with insufficient observations
    obs_counts = returns_df.notna().sum()
    sufficient = obs_counts[obs_counts >= min_observations].index.tolist()
    dropped2 = set(returns_df.columns) - set(sufficient)
    if dropped2:
        print(f"  Dropped {len(dropped2)} tickers (<{min_observations} obs)")
    returns_df = returns_df[sufficient]

    # Drop any remaining NaN rows
    returns_df = returns_df.dropna()

    print(f"  Final dataset: {returns_df.shape[0]} periods x {returns_df.shape[1]} tickers")
    print(f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")

    return returns_df


def main():
    parser = argparse.ArgumentParser(
        description="Download full S&P 500 returns for TFM project"
    )
    parser.add_argument(
        "--start", default="2018-01-01", help="Start date (default: 2018-01-01)"
    )
    parser.add_argument(
        "--end", default="2025-12-31", help="End date (default: 2025-12-31)"
    )
    parser.add_argument(
        "--min-obs", type=int, default=400,
        help="Minimum observations per ticker (default: 400)"
    )
    parser.add_argument(
        "--max-missing", type=float, default=5.0,
        help="Max missing data percent (default: 5.0)"
    )
    args = parser.parse_args()

    project_root = find_project_root()
    output_path = project_root / "data" / "returns_sp500_full.csv"

    print("=" * 70)
    print("S&P 500 FULL DATA DOWNLOAD")
    print("=" * 70)

    returns_df = download_sp500(
        start_date=args.start,
        end_date=args.end,
        min_observations=args.min_obs,
        max_missing_pct=args.max_missing,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    returns_df.to_csv(output_path)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")
    print(f"Tickers: {returns_df.shape[1]}")
    print(f"Periods: {returns_df.shape[0]}")

    # Quick sanity check
    ew = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    port_returns = returns_df.values @ ew
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    print(f"\nEqual-weight portfolio (all {returns_df.shape[1]} tickers):")
    print(f"  VaR(95%)  = {var_95:.6f}")
    print(f"  CVaR(95%) = {cvar_95:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
