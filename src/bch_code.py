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


class BCH_Code(object):
    """This is a representation of a BCH-code."""

    def __init__(self, n, h):
        """Initializer."""
        self._n = n
        # self._generator_pol = generator_pol
        self._h = h
        self._indexes_k = None
        self._indexes_n = None

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
        if self._indexes_k is None:
            self.indexes_k = self._generate_indexes_k(self._n, self.h)
        return self._indexes_k

    @indexes_k.setter
    def indexes_k(self, value):
        """Setter for indexes_k."""
        # print("Setting value indexes_k to", value)
        self._indexes_k = value

    @property
    def indexes_n(self):
        """Getter for indexes_n."""
        if self._indexes_n is None:
            self.indexes_n = self._generate_indexes_n(self._n, self.h)
        return self._indexes_n

    @indexes_n.setter
    def indexes_n(self, value):
        """Setter for indexes_n."""
        # print("Setting value indexes_n to", value)
        self._indexes_n = value

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
