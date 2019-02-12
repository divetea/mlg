#! /usr/bin/python3
"""Test module for bch_code.py.

Using pytest to see how much simpler it is.
"""

from mlg.noisy_word_generator import RandGenerator
import numpy as np


def test_random_codeword():
    """Test random initialization for reproducabiltiy.

    Watch out, order of operations matters since numpy is seeded globally!
    This is not a concern since we only need reproducabilty between runs.
    """
    rand1 = RandGenerator(0.9, 2, mu=1)
    codeword1 = rand1.get_val()
    rand2 = RandGenerator(0.9, 2, mu=1)
    codeword2 = rand2.get_val()
    np.testing.assert_allclose(codeword1, codeword2)
