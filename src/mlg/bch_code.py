#! /usr/bin/python3
"""This is a representation of a BCH-code."""

import numpy as np
from functools import partial


def expon_to_int(exponents):
    """Take a list of exponents and make it into an integer."""
    result = 0
    for exp in exponents:
        result += 2 ** exp
    return result


class BCHCode(object):
    """This is a representation of a BCH-code."""

    def __init__(self, n, h, info_bits):
        """Initializer."""
        self._n = n
        self._info_bits = info_bits
        # self._generator_pol = generator_pol
        self._h = h
        self._indexes_k = self._generate_indexes_k(self.n, self.h)
        self._indexes_n = self._generate_indexes_n(self.n, self.h)
        self._gamma = self.indexes_n.shape[1]
        self._roh = self.indexes_k.shape[1]
        self._f_k = self._weight(self.h) / 2
        self._rate = float(self.info_bits) / float(self.n)

    def syndrome(self, word):
        """Calculate the syndrome for a given word."""
        result = np.empty_like(word)
        for i in range(self.n):
            val = np.mod(np.sum(word[self.indexes_k[i]]), 2)
            result[i] = val
        return np.array(result)

    def _generate_indexes_k(self, n, h):
        bin_h = bin(h)[2:]
        first = []
        for i in range(len(bin_h)):
            if int(bin_h[i]):
                first.append(i)
        result = []
        for x in range(n):
            add_mod_x = partial(self._add_mod, x=x)
            result.append(sorted(map(add_mod_x, first)))
        return np.array(result, dtype=np.int16)

    def _add_mod(self, elem, x):
        return (elem + x) % self.n

    def _generate_indexes_n(self, n, h):
        indexes_k = self.indexes_k
        result = []
        for i in range(self.n):
            result.append([])
            for j, index_k in enumerate(indexes_k):
                if i in index_k:
                    result[i].append(j)

        return np.array(result, dtype=np.int16)

    def _weight(self, polynom):
        return bin(polynom).count('1')

    @property
    def n(self):
        """Getter for n."""
        # print("Getting value n")
        return self._n

    @n.setter
    def n(self, value):
        """Setter for n."""
        # print("Setting value n")
        self._n = value

    @property
    def info_bits(self):
        """Getter for info_bits."""
        return self._info_bits

    @info_bits.setter
    def info_bits(self, value):
        """Setter for info_bits."""
        self._info_bits = value

    @property
    def h(self):
        """Getter for h."""
        if isinstance(self._h, str):
            self.h = int(self._h, 2)
        # print("getting h")
        return self._h

    @h.setter
    def h(self, value):
        """Setter for h."""
        if isinstance(value, str):
            value = int(value, 2)
        # print("Setting value h to", value)
        self._h = value

    @property
    def indexes_k(self):
        """Getter for indexes_k."""
        return self._indexes_k

    @indexes_k.setter
    def indexes_k(self, value):
        """Setter for indexes_k."""
        # print("Setting value indexes_k to", value)
        self._indexes_k = value

    @property
    def indexes_n(self):
        """Getter for indexes_n."""
        return self._indexes_n

    @indexes_n.setter
    def indexes_n(self, value):
        """Setter for indexes_n."""
        self._indexes_n = value

    @property
    def gamma(self):
        """Getter for gamma."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Setter for gamma."""
        self._gamma = value

    @property
    def roh(self):
        """Getter for roh."""
        return self._roh

    @roh.setter
    def roh(self, value):
        """Setter for roh."""
        self._roh = value

    @property
    def f_k(self):
        """Getter for f_k."""
        return self._f_k

    @f_k.setter
    def f_k(self, value):
        """Setter for f_k."""
        self._f_k = value

    @property
    def rate(self):
        """Getter for rate."""
        return self._rate

    @rate.setter
    def rate(self, value):
        """Setter for rate."""
        self._rate = value

    # @property
    # def generator_pol(self):
    #     """Getter for generator_pol."""
    #     # print("Getting value generator_pol")
    #     if isinstance(self._generator_pol, str):
    #         self.generator_pol = int(self._generator_pol, 2)
    #     return self._generator_pol
    #
    # @generator_pol.setter
    # def generator_pol(self, value):
    #     """Setter for generator_pol."""
    #     if isinstance(value, str):
    #         value = int(value, 2)
    #     # print("Setting value generator_pol")
    #     self._generator_pol = value
