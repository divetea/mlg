#! /usr/bin/python3
"""This is a script to demo and plot MLG."""

import csv
import math
import sys

# import matplotlib.pyplot as plt
import numpy as np

from noisy_word_generator import RandGenerator
from bch_code import (BCHCode, expon_to_int)
import mlg_hard
import mlg_soft

sigmas = [.5, .6, .9, 1.3]
sigmas = [.8]
# ends = [1, 2, 3, 4, 5, 6, 10, 20, 30, 40]
end = 20
x_bit = 0
num_sim = 100
# h = expon_to_int(
#     [201, 196, 186, 167, 166, 159, 128, 126, 115, 112, 103, 67, 50, 46, 24,
#      18, 0])  # 273
h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 0])
# h = '0b11010001'
# bch = BCHCode(273, h, 191)
bch = BCHCode(63, h, 37)
# bch = BCHCode(15, h, 7)
max_x_bit = int(math.floor(math.log(bch.gamma + 1, 2) + 1))
print("soft quantization to {} bit.".format(max_x_bit))

# for sigma in sigmas:
#     gen = RandGenerator(sigma, bch.n, mu=1)
#     b_m = gen.get_val()
#     count, bins, ignored = plt.hist(b_m, 60, density=True)
# plt.show()

print("-" * 50)
print("HARD:\n\n")
rows = []
results = {}
num_special_err = 0
for sigma in sigmas:
    results[sigma] = {"DV": 0, "correct": 0, "wrong": 0}
    gen = RandGenerator(sigma, bch.n, mu=1)
    for i in range(num_sim):
        b_m = gen.get_val()
        b_h = mlg_hard.decide_hard(b_m)
        num_err = np.size(np.where(b_h != 0)[0])
        decoded = mlg_hard.decode_modulated(b_m, bch, end)
        if decoded is None:
            result = "DV"
            results[sigma]["DV"] += 1
        elif np.array_equal(decoded, np.zeros(bch.n)):
            result = "correct"
            results[sigma]["correct"] += 1
            if (num_err > bch.f_k):
                # print("b_m: {}\ncorrectly decoded eventhough error was {}"
                #       "".format(b_m, num_err))
                num_special_err += 1
        else:
            result = "wrong"
            results[sigma]["wrong"] += 1
        rows.append(
            [b_m.tolist(), num_err, sigma, end, bch.n, bch.h,
             result, bch.gamma, bch.roh])
print(results)
print(num_special_err)

with open('hard.csv', 'w', newline='') as csvfile:
    result_writer = csv.writer(csvfile, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        result_writer.writerow(row)

print("-" * 50)
print("SOFT:\n\n")
num_special_err = 0
rows = []
results = {}
for sigma in sigmas:
    results[sigma] = {"DV": 0, "correct": 0, "wrong": 0}
    gen = RandGenerator(sigma, bch.n, mu=1)
    for i in range(num_sim):
        b_m = gen.get_val()
        b_h = mlg_soft.decide_hard(b_m)
        num_err = np.size(np.where(b_h != 0)[0])
        decoded = mlg_soft.decode_modulated(b_m, bch, end=end, x_bit=x_bit)
        if decoded is None:
            result = "DV"
            results[sigma]["DV"] += 1
        elif np.array_equal(decoded, np.zeros(bch.n)):
            result = "correct"
            results[sigma]["correct"] += 1
            if (num_err > bch.f_k):
                # print("b_m: {}\ncorrectly decoded eventhough error was {}"
                #       "".format(b_m, num_err))
                num_special_err += 1
        else:
            result = "wrong"
            results[sigma]["wrong"] += 1
        rows.append(
            [b_m.tolist(), num_err, sigma, end, bch.n, bch.h,
             result, bch.gamma, bch.roh, x_bit])

print(results)
print(num_special_err)
with open('soft.csv', 'w', newline='') as csvfile:
    result_writer = csv.writer(csvfile, delimiter=' ',
                               quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in rows:
        result_writer.writerow(row)
# word = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])
# word_modulated = np.array(
#     [-1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1])
# mlg_hard.decode_hard(word, bch)
# mlg_soft.decode_modulated(word_modulated, bch)

# graphs:
# WER / Eb/N0 : need
