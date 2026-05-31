#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : logger.py  (Utils - Centralized Logging)
# ============================================================================
#
# Description
# -----------
# File-only logging system for the TFM project. Writes ONLY to files,
# never to console. Each module gets its own log file (logs/{name}.txt),
# overwritten on each run. Thread-safe via Python GIL.
#
# Key Functions
# -------------
# get_logger(name) : Get or create Logger instance for a module.
# Logger.info/debug/warning/error/critical : Standard log levels.
# set_log_directory(path) : Change log output directory.
# clear_all_logs() : Remove all log files.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

class Logger:
    """
    File-only logger. Writes to logs/{name}.txt, overwrites on each run.
    NO console output (guaranteed).
    
    Thread-safe via Python GIL.
        """
    
    _instances: Dict[str, "Logger"] = {}
    _log_dir = "logs"
    _initialized_loggers = set()
    
    def __new__(cls, name: str):
        """Singleton per logger name - thread-safe via GIL"""
        if name not in cls._instances:
            instance = super(Logger, cls).__new__(cls)
            cls._instances[name] = instance
            instance._initialized = False
        return cls._instances[name]
    
    def __init__(self, name: str):
        """Initialize logger if not already done"""
        if self._initialized:
            return
        
        self.name = name
        self._initialized = True
        
        # Create logs directory if it does not exist
        os.makedirs(self._log_dir, exist_ok=True)
        
        # Log file: logs/{name}.txt (overwritten each run)
        self.log_file = os.path.join(self._log_dir, f"{name}.txt")
        
        # Clear file on first logger init (write mode)
        # If multiple loggers created in same run, only the first one clears
        if self.log_file not in self._initialized_loggers:
            try:
                with open(self.log_file, 'w', encoding='utf-8') as f:
                    f.write("===========================================================\n")
                    f.write(f"LOG FILE: {self.name}\n")
                    f.write(f"Started: {datetime.now().isoformat()}\n")
                    f.write(f"===========================================================\n\n")
                self._initialized_loggers.add(self.log_file)
            except Exception as e:
                sys.stderr.write(f"Logger initialization error: {e}\n")
    
    def _write_to_file(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """
        Write message to file with timestamp and optional metadata.
        
        Parameters
        ----------
        level : str
            Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        message : str
            Main message
        extra : dict, optional
            Additional metadata to append 
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted_message = f"[{timestamp}] [{level:8s}] {message}"
        
        # Add extra metadata if provided 
        if extra and isinstance(extra, dict):
            extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
            formatted_message = f"{formatted_message} | {extra_str}"
        
        formatted_message += "\n"
        
        # Append mode after initialization
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
        except Exception as e:
            # If writing fails, try stderr (no visible console log)
            sys.stderr.write(f"[LOGGER_ERROR] {e}\n")
    
    # =================================================================
    # STANDARD LOG LEVELS 
    # =================================================================
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._write_to_file("DEBUG", message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._write_to_file("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._write_to_file("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message"""
        self._write_to_file("ERROR", message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self._write_to_file("CRITICAL", message, extra)
    
    # =================================================================
    # GENERIC LOG 
    # =================================================================
    
    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log with custom level"""
        self._write_to_file(level.upper(), message, extra)
    
    # =================================================================
    # CLASS METHODS 
    # =================================================================
    
    @classmethod
    def set_log_dir(cls, log_dir: str):
        """Change log directory (use before creating loggers)"""
        cls._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    @classmethod
    def get_log_file(cls, name: str) -> str:
        """Get path to log file for module"""
        return os.path.join(cls._log_dir, f"{name}.txt")
    
    @classmethod
    def clear_all_logs(cls):
        """Clear all log files"""
        if os.path.exists(cls._log_dir):
            for file in os.listdir(cls._log_dir):
                if file.endswith('.txt'):
                    try:
                        os.remove(os.path.join(cls._log_dir, file))
                    except Exception as e:
                        sys.stderr.write(f"Error clearing log {file}: {e}\n")
    
    @classmethod
    def get_all_log_files(cls) -> list:
        """Get list of all log files"""
        if os.path.exists(cls._log_dir):
            return [
                os.path.join(cls._log_dir, f)
                for f in os.listdir(cls._log_dir)
                if f.endswith('.txt')
            ]
        return []

# ============================================================================
# GLOBAL LOGGER FACTORY
# ============================================================================

_loggers: Dict[str, Logger] = {}

def get_logger(name: str) -> Logger:
    """
    Get or create logger for module name.
    
    Parameters
    ----------
    name : str
        Module name (e.g., "cvar_computation", "data_loader")
        Can also use __name__ from modules
    
    Returns
    -------
    Logger
        Logger instance writing to logs/{name}.txt
    
    Examples
    --------
    >>> logger = get_logger("my_module")
    >>> logger.info("Starting computation")
    >>> logger.error("Error occurred")
    
    >>> # In actual module:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Module initialized")
    """
    if name not in _loggers:
        _loggers[name] = Logger(name)
    return _loggers[name]

def set_log_directory(log_dir: str):
    """Set custom log directory for all loggers"""
    Logger.set_log_dir(log_dir)

def get_log_directory() -> str:
    """Get current log directory"""
    return Logger._log_dir

def clear_all_logs():
    """Clear all log files"""
    Logger.clear_all_logs()

# ============================================================================
# QUICK REFERENCE & USAGE EXAMPLES
# ============================================================================
