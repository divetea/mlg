#! /usr/bin/python3
"""This is an implementation of soft reliability based MLG decoding."""

import math

import numpy as np
import copy

from mlg.mlg_hard import decide_hard


def decode_modulated(word, code, end=1000, x_bit=0):
    """Decode a noisy word and return the corrected codeword or None."""
    if not x_bit:
        x_bit = int(math.floor(math.log(code.gamma + 1, 2) + 1))
    max = 2 ** (x_bit - 1) - 1
    tau = 0
    b_m = copy.deepcopy(word)
    # print("b_m({}): {}".format(tau, b_m))
    b_h = decide_hard(b_m)
    syn = code.syndrome(b_h)
    # print("s({}): {}".format(tau, syn))
    r = quantize(b_m, x_bit)
    # print("r({}): {}".format(tau, r))
    # print("b_h({}): {}".format(tau, b_h))

    while(any(syn) and tau < end):
        e = np.zeros(code.n)
        for j in range(code.n):
            e[j] = np.sum(
                np.logical_xor(syn[code.indexes_n[j]], b_h[j]) * 2 - 1)
        # print("e({}): {}\n\n".format(tau, e))
        tau += 1
        r = np.max(
            np.column_stack((r - e, np.ones(code.n) * (-max))),
            axis=1)
        r = np.min(
            np.column_stack((r, np.ones(code.n) * (max))),
            axis=1)
        b_h = np.where(r >= 0, 0, 1)
        # print("r({}): {}".format(tau, r))
        # print("b_h({}): {}".format(tau, b_h))
        syn = code.syndrome(b_h)
        # print("s({}): {}".format(tau, syn))

    if any(syn):
        return None
    return (b_h, tau)


def quantize(word, x):
    """Quantize all values in word with a precision of x bits."""
    max = 2 ** (x - 1) - 1
    rounded = np.around(word * max)
    # print(rounded)
    result = np.where(rounded > max, max, rounded)
    # print(result)
    result = np.where(result < -max, -max, result)
    # print(result)
    return result
