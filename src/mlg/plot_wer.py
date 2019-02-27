#! /usr/bin/python3
"""This is a script to plot WER."""

import matplotlib.pyplot as plt
import pandas as pd
from glob import glob


# Make the graphs a bit prettier
# pd.set_option('display.mpl_style', 'default')
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.family'] = 'sans-serif'


def plot_wer(noises, wer):
    """Plot the WER of a single max_iter/code."""
    plt.show()


# max_iter, noise, sigma, num_sim, DV, correct, wrong, WER,
# corrected_big_errors
names = ["max_iter", "noise", "sigma", "num_sim", "elapsed_time", "DV",
         "correct", "wrong", "WER", "corrected_big_errors"]
hard_df = pd.read_csv(glob("*hard_cummu.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)

soft_df = pd.read_csv(glob("*soft_cummu.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)

plt.figure()
plt.title("Hard")
plt.xlabel('E_b / N_0')
plt.ylabel('WER')
plt.yscale("log")
hard_df.set_index('noise', inplace=True)
hard_df.groupby('max_iter')['WER'].plot(legend=True, marker=".")

plt.figure()
plt.title("Soft")
plt.xlabel('E_b / N_0')
plt.ylabel('WER')
plt.yscale("log")
soft_df.set_index('noise', inplace=True)
soft_df.groupby('max_iter')['WER'].plot(legend=True, marker=".")

plt.figure()
plt.title("Soft")
plt.xlabel('E_b / N_0')
plt.ylabel('WER')
plt.yscale("log")
soft_df.loc[soft_df['max_iter'] == 10]['WER'].plot(legend=True, marker=".")
hard_df.loc[soft_df['max_iter'] == 10]['WER'].plot(legend=True, marker=".")

plt.show()
