#! /usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from codeword_generator import RandGenerator


sigmas = [.5, .6, .9]

for sigma in sigmas:
    gen = RandGenerator(sigma, 1000000, mu=1)
    b_m = gen.get_val()
    count, bins, ignored = plt.hist(b_m, 60, density=True)

plt.show()

# klassische Dekodierung hard, soft
# hard/soft MLG
