#! /usr/bin/python3
"""Test module for mlg_hard.py."""

import unittest

import numpy as np

from mlg.bch_code import BCHCode
from mlg.noisy_word_generator import RandGenerator
from mlg.mlg_hard import (
    decide_hard,
    decode_modulated,
    decode_hard,
    _init_r)


class TestDecodeModulated(unittest.TestCase):
    """Test case for decode_modulated."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001', 7)

    def test_ones_codeword(self):
        """Test correct decoding of a ones codeword (zeroes modulated)."""
        codeword = np.ones(self.code.n)
        result = decode_modulated(codeword, self.code)
        np.testing.assert_equal(result, np.zeros(self.code.n))

    def test_correct_decode(self):
        """Test correct decoding of a random codeword."""
        gen = RandGenerator(0.8, 15, mu=1)
        codeword = gen.get_val()
        result = decode_modulated(codeword, self.code)
        np.testing.assert_equal(result, np.zeros(self.code.n))

    def test_wrong_decode(self):
        """Test wrong correction of a random codeword."""
        gen = RandGenerator(0.4, 15, mu=0)
        codeword = gen.get_val()
        result = decode_modulated(codeword, self.code)
        np.testing.assert_equal(
            result, np.array([0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0]))


class TestDecodeHard(unittest.TestCase):
    """Test case for decode_hard."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001', 7)

    def test_init_r(self):
        """Test private method _init_r."""
        word = np.array([0, 1, 0])
        result = _init_r(word, 3)
        np.testing.assert_array_equal(result, np.array([3, -3, 3]))

    def test_zero_codeword(self):
        """Test correct decoding of a zero codeword."""
        codeword = np.zeros(shape=self.code.n)
        result = decode_hard(codeword, self.code, end=10)
        np.testing.assert_array_equal(result, codeword)
        # self.assertEqual(codeword.tolist(), [0 for _ in range(self.code.n)])

    def test_small_error_word(self):
        """Test correction of a single error."""
        word = np.array(
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = decode_hard(word, self.code, end=10)
        np.testing.assert_array_equal(result, np.zeros(self.code.n))


class TestDecideHard(unittest.TestCase):
    """Test case for decide_hard."""

    def test_ones_codeword(self):
        """Test a codeword with all ones to be decided to all zeroes."""
        codeword = np.ones(5)
        decided = decide_hard(codeword)
        np.testing.assert_array_equal(decided, np.zeros(5))


if __name__ == '__main__':
    unittest.main()
