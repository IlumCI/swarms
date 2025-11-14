"""
Controller modules for maintaining persona consistency and style.

Includes value gate and style adapter for preventing identity drift.
"""

from .value_gate import ValueGate
from .style_adapter import StyleAdapter

__all__ = ["ValueGate", "StyleAdapter"]
