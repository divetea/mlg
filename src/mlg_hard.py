#! /usr/bin/python3
"""This is an implementation of hard reliability based MLG decoding."""

import numpy as np


def decide_hard(b_m):
    """Decide each value in b_m to be either 1 or 0."""
    result = []
    for val in b_m:
        result.append(val < 0)
    return np.array(result)


def decode_modulated(b_m, code):
    """Decode a noisy word and return the corrected codeword or None."""
    pass


def decode_hard(b_h, code, end=5):
    """Decode a hard decided word and return the corrected codeword or None."""
    tau = 0
    syn = syndrome(b_h, code)
    r = _init_r(b_h, code.gamma)
    while(any(syn) or tau == end):
        l_ex = 0
        syn = syndrome(b_h, code)
        end += 1
    return b_h


def syndrome(word, code):
    """Calculate the syndrome for a given word."""
    result = np.empty_like(word)
    for i in range(code.n):
        val = np.mod(np.sum(word[code.indexes_k[i]]), 2)
        result[i] = val
    return np.array(result)


def _init_r(word, gamma):
    result = np.where(word == 0, gamma, -gamma)
    return result
