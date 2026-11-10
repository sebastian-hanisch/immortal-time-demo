"""Gemeinsame Fixtures: das Handbeispiel (sechs Container), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))   # measure_immortal_time.py


@pytest.fixture
def six():
    """Sechs Container (Lebenszeit T, Behandlungsbeginn S oder None = nie), von Hand gerechnet (tests/test_model.py): naiv sind Container 1, 3, 5, 6 behandelt
    (Mittel 6,0 gegen 4,5), die Landmarke 3,5 behält 3, 4, 5, 6 mit 7,0 gegen 7,0."""
    return ((2, 1), (3, 5), (4, Fraction(5, 2)), (6, None), (8, 7), (10, 3))
