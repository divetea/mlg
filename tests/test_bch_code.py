#! /usr/bin/python3
"""Test module for bch_code.py."""

import unittest
from src.bch_code import (BCHCode, expon_to_int)


class TestExponToInt(unittest.TestCase):
    """Test case for the expon_to_int method."""

    def test_3210(self):
        """Test expon_to_int()."""
        exponents = [3, 2, 1, 0]
        self.assertEqual(expon_to_int(exponents=exponents), 15)


class TestBCHCode(unittest.TestCase):
    """Test case for the BCHCode object."""

    def test_zero(self):
        """Test BCHCode(0, 0)."""
        bch = BCHCode(0, 0)
        self.assertEqual(bch.indexes_k.tolist(), [])
        self.assertEqual(bch.indexes_n.tolist(), [])

    def test_small1(self):
        """Test BCHCode(3, 2)."""
        bch = BCHCode(3, 2)
        self.assertEqual(bch.indexes_k.tolist(), [[0], [1], [2]])
        self.assertEqual(bch.indexes_n.tolist(), [[0], [1], [2]])

    def test_small2(self):
        """Test BCHCode(4, 3)."""
        bch = BCHCode(4, 3)
        self.assertEqual(
            bch.indexes_k.tolist(),
            [[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertEqual(
            bch.indexes_n.tolist(),
            [[0, 3], [0, 1], [1, 2], [2, 3]])

    def test_construct_with_str(self):
        """Test BCHCode(4, '0b0011')."""
        bch = BCHCode(4, '0b0011')
        self.assertEqual(
            bch.indexes_k.tolist(),
            [[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertEqual(
            bch.indexes_n.tolist(),
            [[0, 3], [0, 1], [1, 2], [2, 3]])


if __name__ == '__main__':
    unittest.main()
