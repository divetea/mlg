#! /usr/bin/python3
"""This is a script to demo and plot MLG."""

# import matplotlib.pyplot as plt
import numpy as np

from noisy_word_generator import RandGenerator
from bch_code import (BCHCode, expon_to_int)
import mlg_hard
import mlg_soft

sigmas = [.5, .6, .9, 1.3]
# sigmas = [.9]
# ends = [1, 2, 3, 4, 5, 6, 10, 20, 30, 40]
end = 10
num_sim = 2000
h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 1])
h = '0b11010001'
bch = BCHCode(15, h)

# for sigma in sigmas:
#     gen = RandGenerator(sigma, bch.n, mu=1)
#     b_m = gen.get_val()
#     count, bins, ignored = plt.hist(b_m, 60, density=True)
# plt.show()

result = dict()
for sigma in sigmas:
    result[sigma] = {"DV": 0, "correct": 0, "wrong": 0}
    gen = RandGenerator(sigma, bch.n, mu=1)
    for i in range(num_sim):
        b_m = gen.get_val()
        decoded = mlg_hard.decode_modulated(b_m, bch, end)
        if decoded is None:
            result[sigma]["DV"] += 1
        elif np.array_equal(decoded, np.zeros(bch.n)):
            result[sigma]["correct"] += 1
        else:
            result[sigma]["wrong"] += 1
print(result)

result = dict()
for sigma in sigmas:
    result[sigma] = {"DV": 0, "correct": 0, "wrong": 0}
    gen = RandGenerator(sigma, bch.n, mu=1)
    for i in range(num_sim):
        b_m = gen.get_val()
        decoded = mlg_soft.decode_modulated(b_m, bch, end=end)
        if decoded is None:
            result[sigma]["DV"] += 1
        elif np.array_equal(decoded, np.zeros(bch.n)):
            result[sigma]["correct"] += 1
        else:
            result[sigma]["wrong"] += 1
print(result)

word = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
word_modulated = np.array([-1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
# mlg_hard.decode_hard(word, bch)
mlg_soft.decode_modulated(word_modulated, bch)

# soft MLG
