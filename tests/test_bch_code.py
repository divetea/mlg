#! /usr/bin/python3
"""Test module for bch_code.py."""

import unittest

import numpy as np

from mlg.bch_code import (BCHCode, expon_to_int)


class TestExponToInt(unittest.TestCase):
    """Test case for the expon_to_int method."""

    def test_3210(self):
        """Test expon_to_int()."""
        exponents = [3, 2, 1, 0]
        self.assertEqual(expon_to_int(exponents=exponents), 15)


class TestBCHCode(unittest.TestCase):
    """Test case for the BCHCode object."""

    def test_small1(self):
        """Test BCHCode(3, 2, 0)."""
        bch = BCHCode(3, 2, 0)
        self.assertEqual(bch.indexes_k.tolist(), [[0], [1], [2]])
        self.assertEqual(bch.indexes_n.tolist(), [[0], [1], [2]])

    def test_small2(self):
        """Test BCHCode(4, 3, 0)."""
        bch = BCHCode(4, 3, 0)
        self.assertEqual(
            bch.indexes_k.tolist(),
            [[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertEqual(
            bch.indexes_n.tolist(),
            [[0, 3], [0, 1], [1, 2], [2, 3]])

    def test_construct_with_str(self):
        """Test BCHCode(4, '0b0011', 0)."""
        bch = BCHCode(4, '0b0011', 0)
        self.assertEqual(
            bch.indexes_k.tolist(),
            [[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertEqual(
            bch.indexes_n.tolist(),
            [[0, 3], [0, 1], [1, 2], [2, 3]])


class TestSyndrome(unittest.TestCase):
    """Test case for syndrome."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001', 7)

    def test_zeros_word(self):
        """Test syndrome calculation for word with all zeros."""
        word = np.zeros(self.code.n)
        result = self.code.syndrome(word)
        np.testing.assert_array_equal(result, word)

    def test_single_error(self):
        """Test syndrome calculation for a word with a single error."""
        word = np.zeros(self.code.n)
        word[2] = 1
        result = self.code.syndrome(word)
        np.testing.assert_array_equal(
            result, np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))


if __name__ == '__main__':
    unittest.main()
