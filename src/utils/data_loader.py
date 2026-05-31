#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : data_loader.py  (Utils - Market Data Loading)
# ============================================================================
#
# Description
# -----------
# Market data loader with multi-source priority chain:
#   CSV (offline) -> yfinance (full history) -> Alpha Vantage (fallback)
#   -> synthetic (last resort).
# Includes L1-L4 validation, caching with hit/miss tracking, and
# passport integration. Predefined portfolios: sp500_10, tech_5, finance_5.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# OPTIONAL IMPORTS - DETECTED AT MODULE LEVEL
# ============================================================================

_HAS_YFINANCE = False
try:
    import yfinance as yf
    _HAS_YFINANCE = True
except ImportError:
    pass

_HAS_ALPHA_VANTAGE = False
try:
    import alpha_vantage
    from alpha_vantage.timeseries import TimeSeries
    _HAS_ALPHA_VANTAGE = True
except ImportError:
    pass

# ============================================================================
# CONFIGURATION
# ============================================================================

ALPHA_VANTAGE_API_KEY = "GMO57YW0NHT4NNER"

# ============================================================================
# FALLBACK CLASSES
# ============================================================================

class DummyLogger:
    """Fallback logger when utils.logger not available."""
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

class DummyPassport:
    """Fallback passport when utils.passport_orchestrator not available."""
    passport_id = "dummy_passport_000"

class DummyOrchestrator:
    """Fallback orchestrator when utils.passport_orchestrator not available."""
    def assign_passport(self, data, source=None, metadata=None):
        return DummyPassport()
    def create_checkpoint(self, passport, name, data, metadata=None):
        return {'checkpoint_id': f'checkpoint_{name}', 'passport': passport}

# ============================================================================
# IMPORTS - WITH SAFE FALLBACK SUPPORT
# ============================================================================

try:
    from utils.logger import get_logger as get_logger_real
    def get_logger(name=None):
        if name is None:
            name = 'data_loader'
        try:
            return get_logger_real(name)
        except TypeError:
            return DummyLogger()
except ImportError:
    def get_logger(name=None):
        return DummyLogger()

try:
    from utils.passport_orchestrator import get_orchestrator as get_orchestrator_real
    def get_orchestrator():
        try:
            return get_orchestrator_real()
        except Exception:
            return DummyOrchestrator()
except ImportError:
    def get_orchestrator():
        return DummyOrchestrator()

# ============================================================================
# CONSTANTS & PREDEFINED PORTFOLIOS
# ============================================================================

PORTFOLIOS = {
    'sp500_10': {
        'name': 'S&P 500 Top 10',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT'],
        'description': '10 largest S&P 500 companies'
    },
    'tech_5': {
        'name': 'Technology Leaders',
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
        'description': 'Tech sector leaders'
    },
    'finance_5': {
        'name': 'Financial Sector',
        'tickers': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'description': 'Major financial institutions'
    }
}

MIN_PERIODS = 2
MAX_PERIODS = 100000
MIN_ASSETS = 2
MAX_ASSETS = 1000

# ============================================================================
# VALIDATION
# ============================================================================

def _validate_returns_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate returns DataFrame (LEVEL 1-4 validation)."""

    # LEVEL 1: Type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"returns_df must be pd.DataFrame, got {type(df).__name__}")

    # LEVEL 2: Shape
    if df.empty:
        raise ValueError("returns_df is empty")
    if len(df) < MIN_PERIODS:
        raise ValueError(f"returns_df must have at least {MIN_PERIODS} rows, got {len(df)}")
    if len(df) > MAX_PERIODS:
        raise ValueError(f"returns_df exceeds {MAX_PERIODS} rows, got {len(df)}")
    if len(df.columns) < MIN_ASSETS:
        raise ValueError(f"returns_df must have at least {MIN_ASSETS} assets, got {len(df.columns)}")
    if len(df.columns) > MAX_ASSETS:
        raise ValueError(f"returns_df exceeds {MAX_ASSETS} assets, got {len(df.columns)}")

    # LEVEL 4: Data quality
    if df.isna().any().any():
        n_nan = df.isna().sum().sum()
        pct = 100 * n_nan / df.size
        raise ValueError(f"returns_df contains {n_nan} NaN values ({pct:.2f}%)")
    if np.any(np.isinf(df.values)):
        n_inf = np.sum(np.isinf(df.values))
        raise ValueError(f"returns_df contains {n_inf} infinite values")

    return df

# ============================================================================
# DATA SOURCE 1: CSV (highest priority - instant, offline)
# ============================================================================

def load_returns_from_csv(
    csv_path: str,
    tickers: List[str],
    start_date: str,
    end_date: str,
    passport: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Load returns from a pre-downloaded CSV file.

    The CSV must have a date index and ticker columns with daily returns.

    Args:
        csv_path: Path to CSV file (absolute or relative to CWD)
        tickers: List of required ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        passport: Optional passport for tracking

    Returns:
        DataFrame of returns, or None if CSV not available/invalid
    """
    logger = get_logger()

    csv_file = Path(csv_path)
    if not csv_file.exists():
        logger.info(f"CSV not found: {csv_file}")
        return None

    try:
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # Filter date range
        df = df.loc[start_date:end_date]

        # Check all tickers present
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            logger.warning(f"CSV missing tickers: {missing}")
            return None

        # Select only requested tickers in order
        df = df[tickers]

        if len(df) < MIN_PERIODS:
            logger.warning(f"CSV has only {len(df)} periods after date filter (min={MIN_PERIODS})")
            return None

        # Handle NaN from holidays/missing days
        if df.isna().any().any():
            n_before = df.isna().sum().sum()
            df = df.ffill().bfill()
            logger.info(f"CSV: filled {n_before} NaN values (forward/backward fill)")

        df = _validate_returns_dataframe(df)

        logger.info(
            f"CSV loaded successfully:\n"
            f"  File: {csv_file.name}\n"
            f"  Shape: {df.shape}\n"
            f"  Date range: {df.index[0].date()} to {df.index[-1].date()}"
        )
        return df

    except Exception as e:
        logger.warning(f"CSV load failed: {e}")
        return None

# ============================================================================
# DATA SOURCE 2: YFINANCE (primary download - full history, no limits)
# ============================================================================

def load_returns_from_yfinance(
    tickers: List[str],
    start_date: str,
    end_date: str,
    passport: Optional[Any] = None
) -> Optional[pd.DataFrame]:
    """
    Load returns from Yahoo Finance via yfinance.

    Downloads adjusted close prices and computes log returns.
    No API key required. No rate limits. Full date range support.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        passport: Optional passport for tracking

    Returns:
        DataFrame of log returns, or None if download fails
    """
    if not _HAS_YFINANCE:
        return None

    logger = get_logger()

    # Validate tickers
    if not tickers or not isinstance(tickers, list):
        logger.warning("yfinance: invalid tickers input")
        return None

    tickers_clean = [str(t).strip().upper() for t in tickers if t]
    if not tickers_clean:
        logger.warning("yfinance: empty tickers after sanitization")
        return None

    logger.info(f"Downloading {len(tickers_clean)} tickers from Yahoo Finance")
    logger.info(f"Tickers: {tickers_clean}")
    logger.info(f"Date range: {start_date} to {end_date}")

    try:
        # Download adjusted close prices (single batch call)
        raw = yf.download(
            tickers_clean,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False
        )

        logger.info(f"yfinance raw shape: {raw.shape}, columns type: {type(raw.columns).__name__}")

        # -- Extract close prices (robust for all yfinance versions) --
        # yfinance <0.2.40: MultiIndex ('Close','AAPL'), ('Open','AAPL'), ...
        # yfinance >=0.2.40: MultiIndex ('Price','Close','AAPL'), ... or flat
        # Single ticker: flat columns ['Open','High','Low','Close','Volume']
        prices = None

        if isinstance(raw.columns, pd.MultiIndex):
            level0_vals = raw.columns.get_level_values(0).unique().tolist()
            logger.info(f"yfinance MultiIndex level 0: {level0_vals}")

            if 'Close' in level0_vals:
                # yfinance <0.2.40: ('Close', 'AAPL'), ('Open', 'AAPL')
                prices = raw['Close']
            elif 'Price' in level0_vals:
                # yfinance >=0.2.40: ('Price', 'Close', 'AAPL')
                price_block = raw['Price']
                if isinstance(price_block.columns, pd.MultiIndex):
                    prices = price_block['Close']
                elif 'Close' in price_block.columns:
                    prices = price_block[['Close']].rename(columns={'Close': tickers_clean[0]})
                else:
                    prices = price_block
            else:
                # Unknown structure: try selecting by column name pattern
                close_cols = [c for c in raw.columns if 'close' in str(c).lower()]
                if close_cols:
                    prices = raw[close_cols]
                    prices.columns = [c[-1] if isinstance(c, tuple) else c for c in prices.columns]

            # Flatten any remaining MultiIndex columns to ticker names
            if prices is not None and isinstance(prices.columns, pd.MultiIndex):
                prices.columns = prices.columns.get_level_values(-1)

        elif len(tickers_clean) == 1:
            # Single ticker: flat columns
            if 'Close' in raw.columns:
                prices = raw[['Close']].rename(columns={'Close': tickers_clean[0]})
            else:
                prices = raw.iloc[:, :1]
                prices.columns = [tickers_clean[0]]
        else:
            # Flat columns with ticker names (unlikely but handle it)
            prices = raw

        if prices is None or prices.empty:
            logger.warning(f"yfinance: could not extract close prices from columns: {raw.columns.tolist()[:5]}")
            return None

        # Ensure columns are strings (ticker names), not tuples
        prices.columns = [str(c) for c in prices.columns]

        logger.info(f"yfinance close prices extracted: {prices.shape}, columns: {prices.columns.tolist()}")

        # Ensure column order matches requested tickers
        available = [t for t in tickers_clean if t in prices.columns]
        missing = [t for t in tickers_clean if t not in prices.columns]

        if missing:
            logger.warning(f"yfinance: tickers not found: {missing}")
            if len(available) < MIN_ASSETS:
                logger.warning(f"yfinance: only {len(available)} tickers available (min={MIN_ASSETS})")
                return None

        prices = prices[available]

        # Drop rows where all values are NaN (non-trading days)
        prices = prices.dropna(how='all')

        # Forward-fill then back-fill remaining NaNs (holidays per exchange)
        if prices.isna().any().any():
            n_nan = prices.isna().sum().sum()
            prices = prices.ffill().bfill()
            logger.info(f"yfinance: filled {n_nan} NaN price values")

        if len(prices) < 2:
            logger.warning(f"yfinance: insufficient price data ({len(prices)} rows)")
            return None

        logger.info(f"Prices downloaded: {prices.shape}")

        # Compute log returns
        returns_df = np.log(prices / prices.shift(1)).dropna()

        if returns_df.empty:
            logger.warning("yfinance: returns DataFrame empty after log calculation")
            return None

        # Validate
        returns_df = _validate_returns_dataframe(returns_df)

        logger.info(
            f"yfinance loaded successfully:\n"
            f"  Shape: {returns_df.shape}\n"
            f"  Tickers: {len(returns_df.columns)}\n"
            f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}\n"
            f"  Full history: YES"
        )
        return returns_df

    except Exception as e:
        logger.warning(f"yfinance download failed: {e}")
        return None

# ============================================================================
# DATA SOURCE 3: ALPHA VANTAGE (fallback - limited to 100 days free tier)
# ============================================================================

def load_returns_from_alpha_vantage(
    tickers: List[str],
    start_date: str,
    end_date: str,
    passport: Optional[Any] = None
) -> pd.DataFrame:
    """
    Load returns from Alpha Vantage API with 3-second delay (free tier compliant).

    NOTE: Free tier returns only ~100 most recent trading days (compact mode).
    Use yfinance for full date ranges.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        passport: Optional passport for tracking

    Returns:
        DataFrame of log returns

    Raises:
        ImportError: If alpha_vantage package not installed
        ValueError: If tickers invalid or no data returned
    """
    if not _HAS_ALPHA_VANTAGE:
        raise ImportError("alpha_vantage package not installed. Install with: pip install alpha_vantage")

    logger = get_logger()

    # Validate tickers
    if tickers is None:
        raise ValueError("tickers cannot be None")
    if not isinstance(tickers, list):
        raise TypeError(f"tickers must be list, got {type(tickers).__name__}")
    if len(tickers) == 0:
        raise ValueError("tickers list is empty")

    tickers_clean = [str(t).strip().upper() for t in tickers if t]
    if len(tickers_clean) == 0:
        raise ValueError("tickers list contains only empty/whitespace values")

    if len(tickers_clean) != len(tickers):
        logger.warning(f"Sanitized tickers from {len(tickers)} to {len(tickers_clean)}")

    logger.info(f"Loading {len(tickers_clean)} tickers from Alpha Vantage API")
    logger.info(f"Tickers: {tickers_clean}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Rate limit: 3 seconds between requests (free tier compliant)")

    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_prices = {}
        total_tickers = len(tickers_clean)

        for idx, ticker in enumerate(tickers_clean, 1):
            try:
                logger.info(f"  [{idx}/{total_tickers}] Downloading {ticker}...")
                data, meta = ts.get_daily(ticker, outputsize='compact')

                if data is None or data.empty:
                    raise ValueError(f"No data for {ticker}")

                prices = data['4. close'].astype(float)
                prices = prices.sort_index()
                prices = prices[(prices.index >= start_dt) & (prices.index <= end_dt)]

                if prices.empty:
                    logger.warning(
                        f"  No data for {ticker} in range {start_date} to {end_date}. "
                        f"Using all available data"
                    )
                    prices = data['4. close'].astype(float).sort_index()

                all_prices[ticker] = prices
                logger.info(f"  {ticker}: {len(prices)} daily candles")

                if idx < total_tickers:
                    logger.info(f"  Waiting 3 seconds before next request...")
                    time.sleep(3)

            except Exception as e:
                logger.error(f"Failed to get {ticker}: {str(e)}")
                raise

        prices_df = pd.DataFrame(all_prices)
        prices_df = prices_df.sort_index()

        if prices_df.empty or prices_df.shape[0] == 0:
            raise ValueError(
                f"Prices DataFrame is empty after extraction. "
                f"Downloaded {len(tickers_clean)} tickers"
            )

        logger.info(f"Combined prices shape: {prices_df.shape}")

        returns_df = np.log(prices_df / prices_df.shift(1)).dropna()

        if returns_df.empty or returns_df.shape[0] == 0:
            raise ValueError(
                f"Returns DataFrame is empty after log calculation. "
                f"Prices shape: {prices_df.shape}"
            )

        returns_df = _validate_returns_dataframe(returns_df)

        logger.info(
            f"Alpha Vantage loaded successfully:\n"
            f"  Shape: {returns_df.shape}\n"
            f"  Tickers: {len(returns_df.columns)}\n"
            f"  Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}\n"
            f"  NOTE: Free tier (compact mode, 100 latest days)"
        )
        return returns_df

    except Exception as e:
        logger.error(f"load_returns_from_alpha_vantage failed: {str(e)}")
        raise

# ============================================================================
# DATA SOURCE 4: SYNTHETIC (last resort)
# ============================================================================

def generate_synthetic_returns(
    n_periods: int = 252,
    n_assets: int = 5,
    seed: int = 42,
    end_date: str = None,
    tickers: List[str] = None,
    passport: Optional[Any] = None
) -> pd.DataFrame:
    """Generate synthetic returns with realistic characteristics."""

    logger = get_logger()

    try:
        if seed is not None:
            np.random.seed(seed)

        logger.info(f"Generating synthetic returns: n_periods={n_periods}, n_assets={n_assets}")

        if not isinstance(n_periods, int) or n_periods < MIN_PERIODS:
            raise ValueError(f"n_periods must be int >= {MIN_PERIODS}, got {n_periods}")
        if not isinstance(n_assets, int) or n_assets < MIN_ASSETS:
            raise ValueError(f"n_assets must be int >= {MIN_ASSETS}, got {n_assets}")

        cov_matrix = np.eye(n_assets) * 0.01
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                cov_matrix[i, j] = 0.005
                cov_matrix[j, i] = 0.005

        returns_data = np.random.multivariate_normal(
            mean=np.zeros(n_assets),
            cov=cov_matrix,
            size=n_periods
        )

        crash_idx = int(n_periods * 0.2)
        returns_data[crash_idx:crash_idx + 5] -= 0.08

        if tickers is None:
            tickers = [f'Asset_{i}' for i in range(n_assets)]

        if end_date:
            end = pd.to_datetime(end_date)
        else:
            end = pd.Timestamp.now()

        dates = pd.bdate_range(end=end, periods=n_periods).sort_values()

        returns_df = pd.DataFrame(
            returns_data,
            columns=tickers[:n_assets],
            index=dates
        )

        returns_df = _validate_returns_dataframe(returns_df)
        logger.info(f"Synthetic returns generated: shape={returns_df.shape}")

        return returns_df

    except Exception as e:
        logger.error(f"generate_synthetic_returns failed: {str(e)}")
        raise

# ============================================================================
# MAIN DATA LOADER CLASS
# ============================================================================

class DataLoader:
    """
    Data loader for portfolio optimization.

    Loading priority:
        1. CSV file (if csv_path configured) - instant, offline
        2. yfinance (if installed) - full history, no rate limits
        3. Alpha Vantage API (if installed) - 100 days free tier
        4. Synthetic data - last resort fallback
    """

    def __init__(
        self,
        portfolio: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        use_custom_tickers: bool = False,
        cache: bool = True,
        csv_path: Optional[str] = None
    ):
        """
        Initialize DataLoader.

        Args:
            portfolio: Predefined portfolio name (if use_custom_tickers=False)
            tickers: Custom ticker list (if use_custom_tickers=True)
            use_custom_tickers: True=use tickers, False=use portfolio_name
            cache: Enable caching
            csv_path: Path to pre-downloaded CSV with returns
        """
        logger = get_logger()

        # Validate tickers strategy (mutual exclusivity)
        if use_custom_tickers:
            if tickers is None or (isinstance(tickers, list) and len(tickers) == 0):
                raise ValueError(
                    "use_custom_tickers=True but tickers is None or empty. "
                    "Provide non-empty tickers list or set use_custom_tickers=False"
                )
            if not isinstance(tickers, list):
                raise TypeError(f"tickers must be list, got {type(tickers).__name__}")

            self.tickers = [str(t).strip().upper() for t in tickers if t]
            if len(self.tickers) == 0:
                raise ValueError("tickers list contains only empty/whitespace values")

            self.portfolio = None
            self.use_custom_tickers = True

            logger.info(
                f"DataLoader initialized: MODE=CUSTOM_TICKERS, "
                f"n_tickers={len(self.tickers)}, tickers={self.tickers}"
            )
        else:
            if portfolio is None or portfolio not in PORTFOLIOS:
                valid_portfolios = list(PORTFOLIOS.keys())
                raise ValueError(
                    f"use_custom_tickers=False but portfolio '{portfolio}' "
                    f"not in PORTFOLIOS. Valid options: {valid_portfolios}"
                )

            self.portfolio = portfolio
            self.tickers = None
            self.use_custom_tickers = False

            portfolio_tickers = PORTFOLIOS[portfolio]['tickers']
            logger.info(
                f"DataLoader initialized: MODE=PORTFOLIO, "
                f"portfolio='{portfolio}', n_assets={len(portfolio_tickers)}"
            )

        # CSV path
        self.csv_path = csv_path

        # Cache setup
        self.cache = cache if isinstance(cache, bool) else True
        self.cache_dir = Path('cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_hits = 0
        self.cache_misses = 0

        # Log available backends
        backends = []
        if csv_path:
            backends.append(f"CSV({csv_path})")
        if _HAS_YFINANCE:
            backends.append("yfinance")
        if _HAS_ALPHA_VANTAGE:
            backends.append("alpha_vantage")
        backends.append("synthetic")
        logger.info(f"Available backends: {' > '.join(backends)}")

    def _get_cache_key(self, tickers: List[str], start_date: str, end_date: str) -> str:
        """Generate cache key for returns data."""
        ticker_str = '_'.join(sorted(tickers))
        cache_str = f"returns_{ticker_str}_{start_date}_{end_date}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load returns from cache if available."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                logger = get_logger()
                logger.info(f"Cache hit: {cache_key}")
                self.cache_hits += 1
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger = get_logger()
                logger.warning(f"Cache read failed: {e}")

        self.cache_misses += 1
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save returns to cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            data.to_pickle(cache_file)
            logger = get_logger()
            logger.info(f"Cached returns: {cache_key}")
        except Exception as e:
            logger = get_logger()
            logger.warning(f"Cache save failed: {e}")

    def load_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load returns using priority chain: CSV -> yfinance -> Alpha Vantage -> synthetic.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame of daily log returns with DatetimeIndex

        Raises:
            RuntimeError: If all sources fail
        """
        logger = get_logger()

        logger.info("=" * 80)
        logger.info("LOAD_RETURNS (DataLoader.load_returns)")
        logger.info("=" * 80)

        # Determine tickers
        if self.use_custom_tickers:
            if self.tickers is None or len(self.tickers) == 0:
                raise ValueError("use_custom_tickers=True but tickers is empty")
            tickers = self.tickers
            logger.info(f"\n[Mode] CUSTOM TICKERS")
            logger.info(f"  Tickers: {tickers}")
            logger.info(f"  Count: {len(tickers)}")
        else:
            if self.portfolio is None or self.portfolio not in PORTFOLIOS:
                raise ValueError(f"Portfolio '{self.portfolio}' not in PORTFOLIOS")
            tickers = PORTFOLIOS[self.portfolio]['tickers']
            logger.info(f"\n[Mode] PREDEFINED PORTFOLIO")
            logger.info(f"  Portfolio: {self.portfolio}")
            logger.info(f"  Tickers: {tickers}")
            logger.info(f"  Count: {len(tickers)}")

        # Check cache first
        cache_key = self._get_cache_key(tickers, start_date, end_date)
        if self.cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

        logger.info(f"\n[Loading] Date range: {start_date} to {end_date}")

        returns_df = None
        source_used = None

        # -- Priority 1: CSV ------------------------------------------------
        if self.csv_path and returns_df is None:
            logger.info("[Source 1/4] Trying CSV...")
            returns_df = load_returns_from_csv(
                self.csv_path, tickers, start_date, end_date
            )
            if returns_df is not None:
                source_used = f"csv:{Path(self.csv_path).name}"

        # -- Priority 2: yfinance -------------------------------------------
        if returns_df is None:
            logger.info("[Source 2/4] Trying yfinance...")
            if _HAS_YFINANCE:
                returns_df = load_returns_from_yfinance(
                    tickers, start_date, end_date
                )
                if returns_df is not None:
                    source_used = "yfinance"
            else:
                logger.info("  yfinance not installed (pip install yfinance)")

        # -- Priority 3: Alpha Vantage --------------------------------------
        if returns_df is None:
            logger.info("[Source 3/4] Trying Alpha Vantage...")
            if _HAS_ALPHA_VANTAGE:
                try:
                    returns_df = load_returns_from_alpha_vantage(
                        tickers, start_date, end_date
                    )
                    if returns_df is not None:
                        source_used = "alpha_vantage"
                except Exception as e:
                    logger.warning(f"  Alpha Vantage failed: {e}")
            else:
                logger.info("  alpha_vantage not installed")

        # -- Priority 4: Synthetic ------------------------------------------
        if returns_df is None or len(returns_df) < 2:
            logger.info("[Source 4/4] Generating synthetic data (fallback)")
            returns_df = generate_synthetic_returns(
                n_periods=500,
                n_assets=len(tickers),
                seed=42,
                end_date=end_date,
                tickers=tickers
            )
            source_used = "synthetic"

        if returns_df is None or len(returns_df) < 2:
            raise RuntimeError("All data sources failed")

        # Save to cache
        if self.cache:
            self._save_to_cache(cache_key, returns_df)

        logger.info(f"\n[Result] Source: {source_used}")
        logger.info(f"  Shape: {returns_df.shape}")
        logger.info(f"  Range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
        logger.info(f"\nCache stats: hits={self.cache_hits}, misses={self.cache_misses}")

        # Store source for external access
        self.last_source = source_used

        return returns_df
