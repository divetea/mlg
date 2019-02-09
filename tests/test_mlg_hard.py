#! /usr/bin/python3
"""Test module for mlg_hard.py."""

import unittest

import numpy as np

from src.mlg_hard import (
    decide_hard,
    decode_modulated,
    decode_hard,
    syndrome)
from src.bch_code import BCHCode


@unittest.skip("not implemented")
class TestDecodeModulated(unittest.TestCase):
    """Test case for decode_modulated."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001')

    def test_ones_codeword(self):
        """Test correct decoding of a ones codeword (zeroes modulated)."""
        codeword = np.ones(shape=self.code.n)
        result = decode_modulated(codeword)
        np.testing.assert_equal(result, codeword)
        # self.assertEqual(codeword.tolist(), [0 for _ in range(self.code.n)])


class TestDecodeHard(unittest.TestCase):
    """Test case for decode_hard."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001')

    def test_zero_codeword(self):
        """Test correct decoding of a zero codeword."""
        codeword = np.zeros(shape=self.code.n)
        result = decode_hard(codeword, self.code)
        np.testing.assert_array_equal(result, codeword)
        # self.assertEqual(codeword.tolist(), [0 for _ in range(self.code.n)])

    @unittest.skip("not implemented")
    def test_small_error_word(self):
        """Test correction of a single error."""
        word = np.array(
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = decode_hard(word, self.code)
        np.testing.assert_array_equal(result, np.zeros(self.code.n))


class TestDecideHard(unittest.TestCase):
    """Test case for decide_hard."""

    def test_ones_codeword(self):
        """Test a codeword with all ones to be decided to all zeroes."""
        codeword = np.ones(5)
        decided = decide_hard(codeword)
        np.testing.assert_array_equal(decided, np.zeros(5))


class TestSyndrome(unittest.TestCase):
    """Test case for syndrome."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001')

    def test_zeros_word(self):
        """Test syndrome calculation for word with all zeros."""
        word = np.zeros(self.code.n)
        result = syndrome(word, self.code)
        np.testing.assert_array_equal(result, word)

    def test_single_error(self):
        """Test syndrome calculation for a word with a single error."""
        word = np.zeros(self.code.n)
        word[2] = 1
        result = syndrome(word, self.code)
        np.testing.assert_array_equal(
            result, np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
