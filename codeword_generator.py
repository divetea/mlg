#! /usr/bin/python3

import numpy as np


class RandGenerator(object):
    def __init__(self, sigma, size, mu=0):
        self.sigma = sigma
        self.size = size
        self.mu = mu
        # make experiments reproducable
        np.random.seed(0)

    def get_val(self):
        return np.random.normal(self.mu, self.sigma, self.size)
