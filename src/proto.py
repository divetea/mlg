#! /usr/bin/python3
"""This is a script to demo and plot MLG."""

import matplotlib.pyplot as plt
import numpy as np

from noisy_word_generator import RandGenerator
from bch_code import (BCHCode, expon_to_int)
import mlg_hard

sigmas = [.5, .6, .9]

for sigma in sigmas:
    gen = RandGenerator(sigma, 63, mu=1)
    b_m = gen.get_val()
    count, bins, ignored = plt.hist(b_m, 60, density=True)
# plt.show()

h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 1])
bch = BCHCode(63, h)

bch = BCHCode(15, '0b11010001')
word = np.zeros(15)
word[2] = 1
print(bch.h)
print(bch.indexes_k)
print(bch.indexes_n)
print(mlg_hard.syndrome(word, bch))
print(mlg_hard._init_r([0, 1, 1], 3))

# klassische Dekodierung hard, soft
# hard/soft MLG
