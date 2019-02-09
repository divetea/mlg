#! /usr/bin/python3
"""This is a script to demo and plot MLG."""

# import matplotlib.pyplot as plt
import numpy as np

from noisy_word_generator import RandGenerator
from bch_code import (BCHCode, expon_to_int)
import mlg_hard

sigmas = [.5, .6, .9]
sigmas = [.7]
# ends = [1, 2, 3, 4, 5, 6, 10, 20, 30, 40]
end = 6
h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 1])
# h = '0b11010001'
bch = BCHCode(63, h)

# for sigma in sigmas:
#     gen = RandGenerator(sigma, bch.n, mu=1)
#     b_m = gen.get_val()
#     count, bins, ignored = plt.hist(b_m, 60, density=True)
# plt.show()

result = dict()
for sigma in sigmas:
    result[sigma] = []
    gen = RandGenerator(sigma, bch.n, mu=1)
    for i in range(100):
        b_m = gen.get_val()
        decoded = mlg_hard.decode_modulated(b_m, bch, end)
        if decoded is None:
            result[sigma].append("DV")
        elif np.array_equal(decoded, np.zeros(bch.n)):
            result[sigma].append("correct")
        else:
            result[sigma].append("wrong")
print(result)
# bch = BCHCode(63, '0b11010001')
# word = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
# mlg_hard.decode_hard(word, bch)

# soft MLG
