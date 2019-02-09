#! /usr/bin/python3
"""Test module for mlg_hard.py."""

import unittest

import numpy as np

from src.bch_code import BCHCode
from src.noisy_word_generator import RandGenerator
from src.mlg_soft import (
    decode_modulated,
    syndrome,
    quantize)


@unittest.skip("not implemented")
class TestDecodeModulated(unittest.TestCase):
    """Test case for decode_modulated."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001')

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


class TestQuantize(unittest.TestCase):
    """Test case for quantize."""

    def setUp(self):
        """Set up example code for all test cases."""
        self.code = BCHCode(15, '0b11010001')

    def test_zeros_word(self):
        """Test syndrome calculation for word with all zeros."""
        word = np.zeros(self.code.n)
        result = quantize(word, 2)
        np.testing.assert_array_equal(result, word)

    def test_word(self):
        """Test syndrome calculation for word with all zeros."""
        word = np.array([-0.2, 1.2, -4, -1.4, 0.1])
        result = quantize(word, 3)
        np.testing.assert_array_equal(
            result,
            np.array([-1, 3, -3, -3, 0]))
