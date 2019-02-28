#! /usr/bin/python3
"""This is a script to plot WER."""

import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import sys


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


names = ["bch_n", "bch_info_bits", "bch_fk", "bch_h", "num_sim", "max_iter",
         "num_err", "noise_db", "sigma", "result", "needed_iter"]
hard_df = pd.read_csv(glob("*hard_raw.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)

soft_df = pd.read_csv(glob("*soft_raw.csv")[0],
                      delimiter=" ",
                      header=None,
                      names=names)

plt.figure()
plt.title("Error density over all possible decoding results.")
hard_df.groupby('result')["num_err"].hist(density=True, histtype="step", linewidth=2)
plt.xlabel('Number of Errors')
plt.ylabel('Density')
plt.legend(("DV", "correct", "wrong"))
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.figure()
plt.title("Corrected errors histogram - Hard")

histo_df = pd.DataFrame()
# wie das aber normierung auf anzahl runs also z.b. 21.000
hard_df.groupby('result')["num_err"].hist(histtype="step", linewidth=2)
# print(histo_df)
# for errors in range(hard_df["num_err"].max() + 1):
#     histo_df.append({"result": "DV", "num_err": hard_df.loc[hard_df.result == "DV"].loc[hard_df.num_err == errors].count()["num_err"] / hard_df.count()["num_err"]}, ignore_index=True)
#     histo_df.append({"result": "correct", "num_err": hard_df.loc[hard_df.result == "correct"].loc[hard_df.num_err == errors].count()["num_err"] / hard_df.count()["num_err"]}, ignore_index=True)
#     histo_df.append({"result": "wrong", "num_err": hard_df.loc[hard_df.result == "wrong"].loc[hard_df.num_err == errors].count()["num_err"] / hard_df.count()["num_err"]}, ignore_index=True)
#
# print(histo_df)
# histo_df.groupby('result').plot()
plt.xlabel('Number of bitflips')
plt.ylabel('')
plt.legend(("DV", "correct", "wrong"))
plt.grid(True, which='both',
         linestyle=grid_linestyle, linewidth=grid_linewidth)


# plt.figure()
# plt.title("Maximum iterations - Hard")
# hard_df.set_index('noise', inplace=True)
# hard_df.groupby('max_iter')['WER'].plot(
#     logy=True, marker=markerstyle, markersize=markersize)
# plt.legend()
# plt.xlabel('E_b / N_0 [dB]')
# plt.ylabel('WER')
# plt.grid(True, which='both',
#          linestyle=grid_linestyle, linewidth=grid_linewidth)
#
# plt.figure()
# plt.title("Maximum iterations - Soft")
# soft_df.set_index('noise', inplace=True)
# soft_df.groupby('max_iter')['WER'].plot(
#     logy=True, marker=markerstyle, markersize=markersize)
# plt.legend()
# plt.xlabel('E_b / N_0 [dB]')
# plt.ylabel('WER')
# plt.grid(True, which='both',
#          linestyle=grid_linestyle, linewidth=grid_linewidth)
#
# plt.figure()
# plt.title("Decoding failure vs. wrong correction - Soft")
# # soft_df.set_index('noise', inplace=True)
# num_sim = soft_df["num_sim"][0.5]
# soft_df.groupby('max_iter')['DV'].apply(
#     pd.DataFrame.div,
#     pd.DataFrame([0]),
#     axis='index',
#     fill_value=num_sim)
# plt.legend()
# plt.xlabel('E_b / N_0 [dB]')
# # plt.ylabel('WER')
#
#
# plt.figure()
# plt.title("WER - soft vs. hard")
#
# soft_df.loc[soft_df['max_iter'] == 10]['WER'].plot(
#     logy=True, marker=markerstyle, markersize=markersize)
# hard_df.loc[hard_df['max_iter'] == 10]['WER'].plot(
#     logy=True, marker=markerstyle, markersize=markersize)
# plt.legend(("soft", "hard"))
# plt.xlabel('E_b / N_0 [dB]')
# plt.ylabel('WER')
# plt.grid(True, which='both',
#          linestyle=grid_linestyle, linewidth=grid_linewidth)

plt.show()
