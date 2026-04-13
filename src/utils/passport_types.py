#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : passport_types.py  (Utils - Passport Data Structures)
# ============================================================================
#
# Description
# -----------
# Dataclasses for the passport tracking system:
#   Seal        - Records function execution (timing, hashes, status).
#   Checkpoint  - Snapshot of data state at a key pipeline point.
#   DataPassport - Complete lineage record aggregating seals/checkpoints.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class Seal:
    """
    Seal left by a function on the passport.
    Records all details of an execution.
    """
    
    # Identification
    function_name: str
    phase: str
    timestamp: str
    
    # Performance
    execution_time_ms: float
    
    # Data hashing
    input_hash: str
    output_hash: str
    
    # Validation
    validation_status: str  # 'valid', 'warning', 'error'
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Data shape
    data_shape: Optional[tuple] = None
    
    # Transformation type
    transformation_type: str = "unknown"
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        # Convert data_shape tuple to list for JSON serialization
        if d['data_shape'] is not None:
            d['data_shape'] = list(d['data_shape'])
        return d
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Seal":
        """Create from dictionary"""
        if data.get('data_shape') is not None:
            data['data_shape'] = tuple(data['data_shape'])
        return cls(**data)

@dataclass
class Checkpoint:
    """
    Checkpoint (snapshot) of data at an important pipeline point.
    Allows reproducing exact state at any moment.
    """
    
    # Identification
    name: str
    timestamp: str
    
    # Data snapshot
    data_hash: str
    data_summary: str
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Additional information
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary"""
        return cls(**data)

@dataclass
class DataPassport:
    """
    Data passport: complete record of a data item journey through the system.
    Documents origin, transformations, validations, and final results.
    """
    
    # Global identification
    passport_id: str
    
    # Data type
    data_type: str
    
    # Origin
    source: str
    
    # Timestamps
    created_at: str
    
    # Creation hash (for change detection)
    creation_hash: str
    
    # All seals (function executions)
    seals: List[Seal] = field(default_factory=list)
    
    # Snapshots (checkpoints)
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)
    
    # Global metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_seal(self, seal: Seal) -> "DataPassport":
        """Add seal to passport"""
        self.seals.append(seal)
        return self
    
    def add_checkpoint(self, checkpoint: Checkpoint) -> "DataPassport":
        """Add checkpoint to passport"""
        self.checkpoints[checkpoint.name] = checkpoint
        return self
    
    def get_seals_for_phase(self, phase: str) -> List[Seal]:
        """Get all seals for a specific phase"""
        return [s for s in self.seals if s.phase == phase]
    
    def get_seals_for_function(self, function_name: str) -> List[Seal]:
        """Get all seals for a specific function"""
        return [s for s in self.seals if s.function_name == function_name]
    
    def get_journey_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of all events (seals + checkpoints)"""
        events = []
        
        # Add seals
        for seal in self.seals:
            events.append({
                'timestamp': seal.timestamp,
                'type': 'seal',
                'function': seal.function_name,
                'phase': seal.phase,
                'status': seal.validation_status,
                'duration_ms': seal.execution_time_ms,
            })
        
        # Add checkpoints
        for name, checkpoint in self.checkpoints.items():
            events.append({
                'timestamp': checkpoint.timestamp,
                'type': 'checkpoint',
                'name': name,
                'data_hash': checkpoint.data_hash,
            })
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def get_total_execution_time_ms(self) -> float:
        """Get total execution time across all seals"""
        return sum(seal.execution_time_ms for seal in self.seals)
    
    def get_validation_status(self) -> str:
        """Get overall validation status"""
        if not self.seals:
            return "no_seals"
        
        statuses = [seal.validation_status for seal in self.seals]
        
        if 'error' in statuses:
            return 'error'
        elif 'warning' in statuses:
            return 'warning'
        else:
            return 'valid'
    
    def get_all_warnings(self) -> List[str]:
        """Get all warnings from all seals"""
        warnings = []
        for seal in self.seals:
            warnings.extend(seal.warnings)
        return warnings
    
    def get_all_errors(self) -> List[str]:
        """Get all errors from all seals"""
        errors = []
        for seal in self.seals:
            errors.extend(seal.errors)
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'passport_id': self.passport_id,
            'data_type': self.data_type,
            'source': self.source,
            'created_at': self.created_at,
            'creation_hash': self.creation_hash,
            'seals': [seal.to_dict() for seal in self.seals],
            'checkpoints': {
                name: checkpoint.to_dict()
                for name, checkpoint in self.checkpoints.items()
            },
            'metadata': self.metadata,
            'statistics': {
                'total_seals': len(self.seals),
                'total_checkpoints': len(self.checkpoints),
                'total_execution_time_ms': self.get_total_execution_time_ms(),
                'overall_validation_status': self.get_validation_status(),
                'total_warnings': len(self.get_all_warnings()),
                'total_errors': len(self.get_all_errors()),
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPassport":
        """Create from dictionary"""
        seals = [Seal.from_dict(s) for s in data.get('seals', [])]
        checkpoints = {
            name: Checkpoint.from_dict(c)
            for name, c in data.get('checkpoints', {}).items()
        }
        
        return cls(
            passport_id=data['passport_id'],
            data_type=data['data_type'],
            source=data['source'],
            created_at=data['created_at'],
            creation_hash=data['creation_hash'],
            seals=seals,
            checkpoints=checkpoints,
            metadata=data.get('metadata', {}),
        )

# ============================================================================
# SERIALIZATION HELPERS
# ============================================================================

def save_passport_to_file(passport: DataPassport, filepath: str) -> None:
    """Save passport to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(passport.to_json())

def load_passport_from_file(filepath: str) -> DataPassport:
    """Load passport from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return DataPassport.from_dict(data)
