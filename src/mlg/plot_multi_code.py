#! /usr/bin/python3
"""This is a script to plot WER."""

import sys
from collections import OrderedDict

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
print("new_data")


def plot_wer(noises, wer):
    """Plot the WER of a single max_iter/code."""
    plt.show()


# max_iter, noise, sigma, num_sim, elapsed_time, DV, correct, wrong, WER,
# corrected_big_errors
names = ["max_iter", "noise", "sigma", "num_sim", "elapsed_time", "DV",
         "correct", "wrong", "WER", "corrected_big_errors"]

hard_dfs = OrderedDict()
soft_dfs = OrderedDict()
for hard_file in glob("*hard_cummu.csv"):
    code = hard_file[0:hard_file.find(')') + 1]
    hard_dfs[code] = pd.read_csv(
        hard_file,
        delimiter=" ",
        header=None,
        names=names)
    soft_file = hard_file[:-14] + "soft_cummu.csv"
    soft_dfs[code] = pd.read_csv(
        soft_file,
        delimiter=" ",
        header=None,
        names=names)

for code, hard_df in hard_dfs.items():
    hard_df.set_index('noise', inplace=True)
    hard_df["rel_DV"] = hard_df["DV"].divide(
        hard_df["DV"] + hard_df["wrong"])
for code, soft_df in soft_dfs.items():
    soft_df.set_index('noise', inplace=True)
    soft_df["rel_DV"] = soft_df["DV"].divide(
        soft_df["DV"] + soft_df["wrong"])

plt.figure()
plt.title("WER - soft")
for code, soft_df in soft_dfs.items():
    soft_df['WER'].plot(
        logy=True, marker=markerstyle, markersize=markersize)
plt.legend([key + "BCH" for key in soft_dfs.keys()])
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('WER')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("WER - hard")
for code, hard_df in hard_dfs.items():
    hard_df['WER'].plot(
        logy=True, marker=markerstyle, markersize=markersize)
plt.legend([key + "BCH" for key in hard_dfs.keys()])
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('WER')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("TIME - soft")
for code, soft_df in soft_dfs.items():
    soft_elapsed = soft_df['elapsed_time'] * 1000.0 / soft_df["num_sim"].iloc[0]
    soft_elapsed.plot(marker=markerstyle, markersize=markersize)
plt.legend([key + "BCH" for key in soft_dfs.keys()])
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('Simulation time per word [ms]')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("TIME - hard")
for code, hard_df in hard_dfs.items():
    hard_elapsed = hard_df['elapsed_time'] * 1000.0 / hard_df["num_sim"].iloc[0]
    hard_elapsed.plot(marker=markerstyle, markersize=markersize)
plt.legend([key + "BCH" for key in hard_dfs.keys()])
plt.xlabel('E_b / N_0 [dB]')
plt.ylabel('Simulation time per word [ms]')
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.show()
