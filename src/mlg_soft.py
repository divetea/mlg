#! /usr/bin/python3
"""This is an implementation of soft reliability based MLG decoding."""

import numpy as np
import copy


def decode_modulated(word, code, x_bit, end=1000):
    """Decode a noisy word and return the corrected codeword or None."""
    b_m = copy.deepcopy(word)
    tau = 0
    syn = syndrome(b_m, code)
    # print("s_{}: {}".format(tau, syn))
    b_q = quantize(word, x_bit)
    r = b_q
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


def quantize(word, x):
    """Quantize all values in word with a precision of x bits."""
    max = 2 ** (x - 1) - 1
    rounded = np.around(word * max)
    print(rounded)
    result = np.where(rounded > max, max, rounded)
    print(result)
    result = np.where(result < -max, -max, result)
    print(result)
    return result
