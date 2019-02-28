#! /usr/bin/python3
"""This is a script to plot WER."""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from glob import glob


# Make the graphs a bit prettier
# pd.set_option('display.mpl_style', 'default')
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.family'] = 'sans-serif'
grid_linestyle = "--"
grid_linewidth = 0.5
markerstyle = "+"
markersize = 6


def plot_wer(noises, wer):
    """Plot the WER of a single max_iter/code."""
    plt.show()


# max_iter, noise, sigma, num_sim, DV, correct, wrong, WER,
# corrected_big_errors

names = ["max_iter", "noise", "sigma", "num_sim", "DV",
         "correct", "wrong", "WER", "corrected_big_errors"]
hard_df = pd.read_csv(glob("*hard_cummu.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)
hard_df.set_index('noise', inplace=True)

soft_df = pd.read_csv(glob("*soft_cummu.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)
soft_df.set_index('noise', inplace=True)

num_sim = int(hard_df["num_sim"].iloc[0])
for key in ["DV", "correct", "wrong", "corrected_big_errors"]:
    hard_df[key] = hard_df[key].div(num_sim)
    soft_df[key] = soft_df[key].div(num_sim)

hard_df["rel_DV"] = hard_df["DV"].divide(
    hard_df["DV"] + hard_df["wrong"])
soft_df["rel_DV"] = soft_df["DV"].divide(
    soft_df["DV"] + soft_df["wrong"])

plt.figure()
plt.title("Maximum iterations - Hard")
hard_df.groupby('max_iter')['WER'].plot(
    logy=True, marker=markerstyle, markersize=markersize)
plt.legend()
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('WER')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("Maximum iterations - Soft")
soft_df.groupby('max_iter')['WER'].plot(
    logy=True, marker=markerstyle, markersize=markersize)
plt.legend()
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('WER')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("Decoding Failure / (Dec. Fail. + Wrong Correction)")
hard_df.loc[hard_df['max_iter'] == 10]['rel_DV'].plot(
    marker=markerstyle, markersize=markersize)
hard_df.loc[hard_df['max_iter'] == 100]['rel_DV'].plot(
    marker=markerstyle, markersize=markersize)
plt.legend(("10 iterations", "100 iterations"))
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('relative Decoding Failure Rate')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("Decoding Failure / Dec. Fail. + Wrong Correction - Soft")
soft_df.loc[soft_df['max_iter'] == 10]['rel_DV'].plot(
    marker=markerstyle, markersize=markersize)
soft_df.loc[soft_df['max_iter'] == 100]['rel_DV'].plot(
    marker=markerstyle, markersize=markersize)
plt.legend(("10 iterations", "100 iterations"))
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('relative Decoding Failure Rate')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.show()
