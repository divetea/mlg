#! /usr/bin/python3
"""Test module for bch_code.py.

Using pytest to see how much simpler it is.
"""

from src.codeword_generator import RandGenerator


def test_random():
    """Test random initialization for reproducabiltiy."""
    rand = RandGenerator(0.9, 2, mu=1)
    assert rand.get_val()[0] == 2.587647111370898
    assert rand.get_val()[1] == 3.016803879281312
