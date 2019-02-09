#! /usr/bin/python3
"""This is an implementation of hard reliability based MLG decoding."""

import numpy as np


def decide_hard(b_m):
    """Decide each value in b_m to be either 1 or 0."""
    result = []
    for val in b_m:
        result.append(val < 0)
    return np.array(result)


def decode_modulated(b_m):
    """Decode a noisy word and return the corrected codeword or None."""
    pass


def decode_hard(b_h):
    """Decode a hard decided word and return the corrected codeword or None."""
    pass
