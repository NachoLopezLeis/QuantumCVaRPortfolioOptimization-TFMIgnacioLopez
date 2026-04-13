#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : config_loader.py  (Utils - Configuration Management)
# ============================================================================
#
# Description
# -----------
# Centralized YAML configuration manager (singleton pattern).
# Loads from config/config.yaml (the single source of truth).
# Supports nested key access with dot notation, caching, and
# phase-specific configuration helpers.
#
# Fix [P3-002]: Previously looked for paths.yaml, phase1.yaml, phase2.yaml
# which do not exist. Now correctly loads config/config.yaml.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .logger import get_logger

logger = get_logger(__name__)


class ConfigError(Exception):
    """Base exception for configuration errors."""


class ConfigNotFoundError(ConfigError):
    """Raised when the configuration file is not found."""


class Config:
    """
    Centralized configuration manager (singleton).
    Loads config/config.yaml and exposes nested values via dot-notation.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.config_data: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_enabled: bool = True
        self._load_config()

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    def _get_config_dir(self) -> Path:
        """Resolve path to the config/ directory from any working directory."""
        candidates = [
            Path(__file__).parent.parent.parent / "config",   # src/../config
            Path(__file__).parent.parent / "config",          # src/config
            Path.cwd() / "config",
            Path.cwd().parent / "config",
        ]
        for p in candidates:
            if (p / "config.yaml").exists():
                return p
        # Fallback — will raise on load
        return Path(__file__).parent.parent.parent / "config"

    def _load_config(self) -> None:
        """Load configuration from config/config.yaml."""
        config_dir = self._get_config_dir()
        config_path = config_dir / "config.yaml"

        if not config_path.exists():
            logger.warning(f"config.yaml not found at {config_path}. "
                           "Configuration will be empty.")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
            if loaded and isinstance(loaded, dict):
                self.config_data = loaded
                logger.info(f"Configuration loaded from {config_path} "
                            f"({len(self.config_data)} top-level keys)")
            else:
                logger.warning("config.yaml parsed but returned no data.")
        except Exception as exc:
            logger.error(f"Error loading config.yaml: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a (possibly nested) config value using dot notation.

        Parameters
        ----------
        key : str
            Dot-separated key path, e.g. 'phase3.benchmarking.enabled'.
        default : Any
            Value to return when key is missing.

        Returns
        -------
        Any
            Config value or default.
        """
        if self._cache_enabled and key in self._cache:
            return self._cache[key]

        parts = key.split(".")
        value: Any = self.config_data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                value = default
                break

        if value is None:
            value = default

        if self._cache_enabled:
            self._cache[key] = value
        return value

    def set(self, key: str, value: Any) -> None:
        """Set a config value (dot notation). Clears cache entry."""
        parts = key.split(".")
        node = self.config_data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        self._cache.pop(key, None)

    def get_all(self) -> Dict[str, Any]:
        """Return a shallow copy of the full config dict."""
        return dict(self.config_data)

    def reload(self, config_path: Optional[Path] = None) -> None:
        """Reload configuration from disk and clear cache."""
        self.config_data = {}
        self._cache = {}
        if config_path is not None:
            try:
                with open(config_path, "r", encoding="utf-8") as fh:
                    loaded = yaml.safe_load(fh)
                if loaded and isinstance(loaded, dict):
                    self.config_data = loaded
                    logger.info(f"Configuration reloaded from {config_path}")
            except Exception as exc:
                logger.error(f"Error reloading config from {config_path}: {exc}")
        else:
            self._load_config()


# ============================================================================
# Singleton helpers
# ============================================================================

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Return the singleton Config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def init_config(config_path: Optional[Path] = None) -> Config:
    """(Re-)initialize and return the singleton Config instance."""
    global _config_instance
    _config_instance = Config.__new__(Config)
    _config_instance._initialized = False
    _config_instance.__init__()
    if config_path is not None:
        _config_instance.reload(config_path)
    return _config_instance


def get_phase1_config() -> Dict[str, Any]:
    return get_config().get("phase1", {})


def get_phase2_config() -> Dict[str, Any]:
    return get_config().get("phase2", {})


def get_phase3_config() -> Dict[str, Any]:
    return get_config().get("phase3", {})


def get_paths_config() -> Dict[str, Any]:
    return get_config().get("paths", {})
