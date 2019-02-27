#! /usr/bin/python3
"""This is a script to simulate all data for WER plot."""

import csv
import sys
import time
from collections import OrderedDict
from functools import partial

import numpy as np

from noisy_word_generator import RandGenerator
from bch_code import (BCHCode, expon_to_int)
from mlg_hard import decide_hard
if len(sys.argv) == 2:
    print("#" * 100, "\n\nUSING HARD DECISION!\n\n")
    import mlg_hard as decoder
    algo = "hard"
else:
    print("#" * 100, "\n\nUSING SOFT DECISION!\n\n")
    import mlg_soft as decoder
    algo = "soft"


def noise_to_sigma(noise, r):
    """Calculate a sigma^2 value for a given E_b/N_0."""
    return 1.0 / (2.0 * r * (10.0 ** (noise / 10.0)))


codes = []
h = '0b11010001'
codes.append(BCHCode(15, h, 7))
h = expon_to_int([31, 15, 7, 3, 1, 0])
codes.append(BCHCode(63, h, 31))
h = expon_to_int([37, 32, 25, 22, 21, 8, 2, 0])
codes.append(BCHCode(63, h, 37))
h = expon_to_int([63, 54, 36, 31, 15, 7, 3, 1, 0])
codes.append(BCHCode(73, h, 45))
h = expon_to_int([61, 47, 30, 23, 11, 5, 2])
codes.append(BCHCode(93, h, 47))
h = expon_to_int([53, 46, 45, 32, 30, 4, 0])
codes.append(BCHCode(105, h, 53))
h = expon_to_int(
    [201, 196, 186, 167, 166, 159, 128, 126, 115, 112, 103, 67, 50, 46, 24,
     18, 0])
codes.append(BCHCode(273, h, 191))

for bch in codes:
    # n l f_k h(x) num_sim max_iter num_err noise[dB] sigma result needed_iter
    filename_raw = "({},{},{})_wer_{}_raw.csv".format(
        bch.n, bch.info_bits, int(bch.f_k), algo)

    # max_iter, noise, sigma, num_sim, DV, correct, wrong, WER,
    # corrected_big_errors
    filename_cumm = "({},{},{})_wer_{}_cummu.csv".format(
        bch.n, bch.info_bits, int(bch.f_k), algo)

    noises = np.linspace(0.5, 3.5, num=21)
    print("simulating for following noise levels:", noises)
    sigmas = list(map(partial(noise_to_sigma, r=bch.rate), noises))

    max_iterations = [3, 4, 5, 8, 10, 15, 20, 100]
    max_iterations = [10]
    num_sim = 1000
    # annahme WER = 1 - (#correct / num_sim)
    print(bch.__dict__)
    for max_iter in max_iterations:
        cummulated_by_sigma = []
        print("_"*70)
        print("max iterations: ", max_iter)
        results = OrderedDict()
        for sigma_counter, sigma in enumerate(sigmas):
            before = time.process_time()
            rows = []
            print("-"*30)
            print("Sigma: ", sigma)
            results[sigma] = OrderedDict([("DV", 0),
                                          ("correct", 0),
                                          ("wrong", 0)])
            gen = RandGenerator(sigma, bch.n, mu=1)
            num_special_err = 0
            for i in range(num_sim):
                b_m = gen.get_val()
                b_h = decide_hard(b_m)
                num_err = np.size(np.where(b_h != 0)[0])
                decoded_tuple = decoder.decode_modulated(b_m, bch, max_iter)
                if decoded_tuple is None:
                    needed_iter = -1
                    result = "DV"
                    results[sigma]["DV"] += 1
                else:
                    decoded, needed_iter = decoded_tuple
                    if np.array_equal(decoded, np.zeros(bch.n)):
                        result = "correct"
                        results[sigma]["correct"] += 1
                        if (num_err > bch.f_k):
                            num_special_err += 1
                    else:
                        result = "wrong"
                        results[sigma]["wrong"] += 1
                rows.append(
                    [bch.n, bch.info_bits, bch.f_k, bch.h, num_sim, max_iter,
                     num_err, noises[sigma_counter], sigma, result,
                     needed_iter])

            results[sigma]["WER"] = 1 - results[sigma]["correct"] / float(
                num_sim)
            results[sigma]["correct_big_errors"] = num_special_err
            print(results[sigma])
            # DV, correct, wrong, WER, correct_big_errors
            cummulated_row = [v for k, v in results[sigma].items()]
            cummulated_row.insert(0, max_iter)
            cummulated_row.insert(1, noises[sigma_counter])
            cummulated_row.insert(2, sigma)
            cummulated_row.insert(3, num_sim)
            after = time.process_time()
            elapsed_time = after - before
            cummulated_row.insert(4, elapsed_time)
            # max_iter, noise, sigma, num_sim, DV, correct, wrong, WER,
            # correct_big_errors
            cummulated_by_sigma.append(cummulated_row)

            with open(filename_raw, 'a', newline='') as csvfile:
                result_writer = csv.writer(
                    csvfile, delimiter=' ', quotechar='|',
                    quoting=csv.QUOTE_MINIMAL)
                for row in rows:
                    result_writer.writerow(row)
        with open(filename_cumm, 'a', newline='') as csvfile:
            result_writer = csv.writer(csvfile, delimiter=' ',
                                       quotechar='|',
                                       quoting=csv.QUOTE_MINIMAL)
            for row in cummulated_by_sigma:
                result_writer.writerow(row)
