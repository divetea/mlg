#! /usr/bin/python3
"""This is a script to demo and plot MLG."""

import matplotlib.pyplot as plt
from noisy_word_generator import RandGenerator
from bch_code import (BCH_Code, expon_to_int)


sigmas = [.5, .6, .9]

for sigma in sigmas:
    gen = RandGenerator(sigma, 63, mu=1)
    b_m = gen.get_val()
    count, bins, ignored = plt.hist(b_m, 60, density=True)
# plt.show()

h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 1])
bch = BCH_Code(63, h)

print(bch.h)
print(bch.indexes_k)
print(bch.indexes_n)

# klassische Dekodierung hard, soft
# hard/soft MLG
