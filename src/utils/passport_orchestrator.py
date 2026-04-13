#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : passport_orchestrator.py  (Utils - Passport Orchestrator)
# ============================================================================
#
# Central orchestrator for data passport tracking.
#
# BUG-001 FIX: dual import pattern (relative + bare) so the module works both
#   when imported as src.utils.passport_orchestrator (pipeline context) and
#   as plain passport_orchestrator (notebook sys.path context).
#
# BUG-008 FIX: PassportOrchestrator now supports a unified master passport
#   chain via link_child() / get_pipeline_chain(). Each module's passport
#   is registered as a child of the master passport so the full data journey
#   can be reconstructed with export_pipeline_chain().
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import uuid
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# BUG-001 FIX: dual import — works both as package module and as bare import
try:
    from .passport_types import Seal, Checkpoint, DataPassport, save_passport_to_file
    from .logger import get_logger
except ImportError:
    from passport_types import Seal, Checkpoint, DataPassport, save_passport_to_file
    from logger import get_logger

logger = get_logger(__name__)


def _hash_data(data: Any) -> str:
    """Compute a 16-char MD5 hash of any data object."""
    try:
        if hasattr(data, 'tobytes'):          # numpy array
            raw = data.tobytes()
        elif isinstance(data, dict):
            raw = json.dumps(data, sort_keys=True, default=str).encode()
        else:
            raw = str(data).encode()
        return hashlib.md5(raw).hexdigest()[:16]
    except Exception:
        return "hash_unavail"


class PassportOrchestrator:
    """
    Central orchestrator for managing data passports.

    BUG-008 FIX: supports a unified pipeline chain.
      orchestrator.set_master(passport)   — register the run's master passport
      orchestrator.link_child(child)      — attach a module passport to master
      orchestrator.export_pipeline_chain(path) — write full chain to JSON
    """

    def __init__(self, enable_tracking: bool = True, log_dir: str = "logs"):
        self.enable_tracking = enable_tracking
        self.log_dir = Path(log_dir)
        self.passports: Dict[str, DataPassport] = {}
        # BUG-008: master passport chain support
        self._master_passport: Optional[DataPassport] = None
        self._child_passports: List[DataPassport] = []

        logger.info(f"PassportOrchestrator initialized (tracking={'ON' if enable_tracking else 'OFF'})")

    # ------------------------------------------------------------------
    # BUG-008: Master passport chain API
    # ------------------------------------------------------------------

    def set_master(self, passport: 'DataPassport') -> None:
        """Register the pipeline's master passport. All subsequent children link to it."""
        self._master_passport = passport
        logger.info(f"Master passport set: {passport.passport_id[:8]}...")

    def link_child(self, child: 'DataPassport') -> None:
        """Attach a module-level passport as a child of the master passport."""
        self._child_passports.append(child)
        if self._master_passport is not None:
            if not hasattr(self._master_passport, 'child_passport_ids'):
                self._master_passport.child_passport_ids = []
            self._master_passport.child_passport_ids.append(child.passport_id)
        logger.debug(f"Child passport linked: {child.passport_id[:8]}... -> master")

    def export_pipeline_chain(self, output_path: str) -> None:
        """
        Export the full data pipeline chain as a structured JSON.

        The chain shows every transformation the data went through:
        master_passport -> child_1 (data loading) -> child_2 (CVaR) ->
        child_3 (QAE) -> child_4 (optimization) -> child_5 (OOS) ...

        This is the primary artifact for audit trail and paper reproducibility.
        """
        chain = {
            "pipeline_chain_version": "2.0",
            "export_timestamp": datetime.now().isoformat(),
            "master": None,
            "stages": [],
        }

        if self._master_passport is not None:
            chain["master"] = {
                "passport_id": self._master_passport.passport_id,
                "created_at": self._master_passport.created_at,
                "data_type": self._master_passport.data_type,
                "source": self._master_passport.source,
                "n_seals": len(getattr(self._master_passport, 'seals', [])),
                "n_checkpoints": len(getattr(self._master_passport, 'checkpoints', [])),
            }

        for i, child in enumerate(self._child_passports):
            seals = getattr(child, 'seals', [])
            checkpoints = getattr(child, 'checkpoints', [])
            stage = {
                "stage_index": i + 1,
                "passport_id": child.passport_id,
                "data_type": getattr(child, 'data_type', 'unknown'),
                "source": getattr(child, 'source', 'unknown'),
                "created_at": getattr(child, 'created_at', ''),
                "seals": [
                    {
                        "function": s.function_name if hasattr(s, 'function_name') else str(s),
                        "phase": s.phase if hasattr(s, 'phase') else '',
                        "timestamp": s.timestamp if hasattr(s, 'timestamp') else '',
                        "status": s.validation_status if hasattr(s, 'validation_status') else '',
                        "execution_time_ms": s.execution_time_ms if hasattr(s, 'execution_time_ms') else 0,
                        "input_hash": s.input_hash if hasattr(s, 'input_hash') else '',
                        "output_hash": s.output_hash if hasattr(s, 'output_hash') else '',
                        "transformation_type": s.transformation_type if hasattr(s, 'transformation_type') else '',
                    }
                    for s in seals
                ],
                "checkpoints": [
                    {
                        "name": c.name if hasattr(c, 'name') else str(c),
                        "timestamp": c.timestamp if hasattr(c, 'timestamp') else '',
                        "data_hash": c.data_hash if hasattr(c, 'data_hash') else '',
                    }
                    for c in checkpoints
                ],
            }
            chain["stages"].append(stage)

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(chain, indent=2, default=str), encoding="utf-8")
        logger.info(f"Pipeline chain exported: {path} ({len(self._child_passports)} stages)")

    # ------------------------------------------------------------------
    # Core passport API
    # ------------------------------------------------------------------

    def assign_passport(
        self,
        data: Any,
        data_type: str = "generic",
        source: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> 'DataPassport':
        """Create and register a new passport for a data object."""
        if not self.enable_tracking:
            return self._dummy_passport(data_type, source)

        passport_id = str(uuid.uuid4())
        data_hash   = _hash_data(data)
        now         = datetime.now().isoformat()

        passport = DataPassport(
            passport_id=passport_id,
            created_at=now,
            data_type=data_type,
            source=source,
            creation_hash=data_hash,
            metadata=metadata or {},
            seals=[],
            checkpoints=[],
        )

        self.passports[passport_id] = passport
        logger.info(f"OK Passport assigned: {passport_id[:8]}... (type={data_type}, source={source})")

        # BUG-008: auto-link to master chain if set
        if self._master_passport is not None and passport_id != getattr(self._master_passport, 'passport_id', None):
            self.link_child(passport)

        return passport

    def seal(
        self,
        passport: 'DataPassport',
        function_name: str,
        phase: str,
        result: Dict[str, Any],
    ) -> 'DataPassport':
        """Seal a passport after a function processes the data."""
        if not self.enable_tracking or passport is None:
            return passport

        try:
            now = datetime.now().isoformat()
            input_hash  = getattr(passport, 'data_hash', 'unknown')
            output_hash = _hash_data(result)

            seal = Seal(
                function_name=function_name,
                phase=phase,
                timestamp=now,
                execution_time_ms=float(result.get('execution_time_ms', 0)),
                input_hash=input_hash,
                output_hash=output_hash,
                validation_status=result.get('validation_status', 'valid'),
                warnings=result.get('warnings', []),
                errors=result.get('errors', []),
                parameters=result.get('parameters', {}),
                data_shape=result.get('data_shape'),
                transformation_type=result.get('transformation_type', 'unknown'),
            )

            if not hasattr(passport, 'seals') or passport.seals is None:
                passport.seals = []
            passport.seals.append(seal)
            passport.data_hash = output_hash

            status_str = seal.validation_status
            if seal.warnings:
                status_str += f" - {'; '.join(seal.warnings[:2])}"
            logger.info(
                f"OK Seal created: {function_name} ({phase}) - "
                f"time={seal.execution_time_ms:.2f}ms, status={status_str}"
            )
        except Exception as exc:
            logger.warning(f"Seal creation failed for {function_name}: {exc}")

        return passport

    def checkpoint(
        self,
        passport: 'DataPassport',
        name: str,
        data: Any,
        metadata: Optional[Dict] = None,
    ) -> 'DataPassport':
        """Create a checkpoint snapshot of data state."""
        if not self.enable_tracking or passport is None:
            return passport

        try:
            cp = Checkpoint(
                name=name,
                timestamp=datetime.now().isoformat(),
                data_hash=_hash_data(data),
                data_summary='',
                metadata=metadata or {},
            )
            if not hasattr(passport, 'checkpoints') or passport.checkpoints is None:
                passport.checkpoints = []
            passport.checkpoints.append(cp)
            logger.debug(f"OK Checkpoint created: {name} - hash={cp.data_hash}...")
        except Exception as exc:
            logger.warning(f"Checkpoint creation failed for {name}: {exc}")

        return passport

    def export_passport(self, passport: 'DataPassport', output_path: str) -> None:
        """Export a single passport to JSON file."""
        if passport is None:
            logger.warning("export_passport: passport is None, skipping")
            return
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_passport_to_file(passport, str(path))
            logger.info(f"Passport exported: {path}")
        except Exception as exc:
            logger.error(f"Passport export failed: {exc}")

    def _dummy_passport(self, data_type: str, source: str) -> Any:
        """Return a lightweight stub when tracking is disabled."""
        class _Stub:
            passport_id = f"disabled_{uuid.uuid4().hex[:8]}"
            seals = []
            checkpoints = []
            data_hash = "tracking_disabled"
            created_at = datetime.now().isoformat()
        stub = _Stub()
        stub.data_type = data_type
        stub.source    = source
        stub.metadata  = {}
        return stub


# ============================================================================
# Singleton
# ============================================================================

_instance: Optional[PassportOrchestrator] = None


def get_orchestrator() -> PassportOrchestrator:
    """Return the singleton PassportOrchestrator."""
    global _instance
    if _instance is None:
        _instance = PassportOrchestrator(enable_tracking=True)
    return _instance


def reset_orchestrator() -> PassportOrchestrator:
    """Reset singleton (useful between runs)."""
    global _instance
    _instance = PassportOrchestrator(enable_tracking=True)
    return _instance
