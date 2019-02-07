#! /usr/bin/python3
"""This is a random number generator with normal distribution.

It is seeded so experiments can be reproduced.
"""

import numpy as np


class RandGenerator(object):
    """This is a random number generator with normal distribution."""

    def __init__(self, sigma, size, mu=0):
        """Initializer."""
        self.sigma = sigma
        self.size = size
        self.mu = mu
        # make experiments reproducable
        np.random.seed(0)

    def get_val(self):
        """Get a new value from the generator."""
        return np.random.normal(self.mu, self.sigma, self.size)
