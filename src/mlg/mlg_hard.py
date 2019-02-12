#! /usr/bin/python3
"""This is an implementation of hard reliability based MLG decoding."""

import numpy as np
import copy


def decide_hard(b_m):
    """Decide each value in b_m to be either 1 or 0."""
    result = []
    for val in b_m:
        result.append(val < 0)
    return np.array(result, dtype=np.int8)


def decode_modulated(b_m, code, end=1000):
    """Decode a noisy word and return the corrected codeword or None."""
    return decode_hard(decide_hard(b_m), code, end)


def decode_hard(word, code, end):
    """Decode a hard decided word and return the corrected codeword or None."""
    b_h = copy.deepcopy(word)
    tau = 0
    syn = syndrome(b_h, code)
    # print("s_{}: {}".format(tau, syn))
    r = _init_r(b_h, code.gamma)
    # print("r_{}: {}".format(tau, r))
    # print("b_{}: {}".format(tau, b_h))
    while(any(syn) and tau <= end):
        e = np.zeros(code.n)
        for j in range(code.n):
            e[j] = np.sum(
                np.logical_xor(syn[code.indexes_n[j]], b_h[j]) * 2 - 1)
        # print("e_{}: {}\n\n".format(tau, e))
        tau += 1
        r = np.max(
            np.column_stack((r - e, np.ones(code.n) * (-code.gamma))),
            axis=1)
        r = np.min(
            np.column_stack((r, np.ones(code.n) * (code.gamma))),
            axis=1)
        b_h = np.where(r >= 0, 0, 1)
        # print("r_{}: {}".format(tau, r))
        # print("b_{}: {}".format(tau, b_h))
        syn = syndrome(b_h, code)
        # print("s_{}: {}".format(tau, syn))

    if any(syn):
        return None
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
