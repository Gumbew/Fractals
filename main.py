import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def calc_Gi(i, j):
    i_log = np.log(i)
    j_log = np.log(j)
    return j_log - i_log


def get_avg_g_i(g_i):
    return np.average(g_i)


def get_avg_g_i_sqr(g_i):
    g_i = [x * x for x in g_i]
    return get_avg_g_i(g_i)


def get_gamma_i(g_i):
    avg_gi = get_avg_g_i(g_i)
    avg_gi_sqr = get_avg_g_i_sqr(g_i)
    return np.sqrt(avg_gi_sqr - avg_gi ** 2)


def get_g_little_i(g_i, g_i_t):
    g_i_avg = get_avg_g_i(g_i)
    gamma = get_gamma_i(g_i)
    return (g_i_t - g_i_avg) / gamma


def build_matrix_c(g):
    g_little = []
    for g_i in g:
        g_little_i = []
        for g_i_t in g_i:
            g_little_i.append(get_g_little_i(g_i, g_i_t))
        g_little.append(g_little_i)
    c = []
    for i in range(len(g_little)):
        c_i = []
        for j in range(len(g_little)):
            vec_mul = np.array(g_little[i]) * np.array(g_little[j])
            c_i.append(np.average(vec_mul))
        c.append(c_i)
    return c


def show_heat_map(C):
    C = np.flipud(C)
    sns.heatmap(C, yticklabels=[i for i in range(len(C[0]), 0, -1)], xticklabels=[i for i in range(1, len(C[0]) + 1)])
    plt.show()


def build_matrix_g(df):
    #gen = df.iterrows()
    #next(gen)
    g = []
    for index, row in df.iterrows():
        # row is row
        g_i = []
        for i in range(1, len(row)):
            g_i.append(calc_Gi(row[i], row[i - 1]))
            # print(row[i], row[i-1])
        g.append(g_i)
    return g


if __name__ == "__main__":
    file_path = f"data{os.path.sep}data_lab_5_without_date.csv"
    df = pd.read_csv(file_path)
    df = df.transpose()

    df_mixed = df.copy()
    df_mixed = df_mixed.sample(frac=1)

    G = build_matrix_g(df)
    G_mixed = build_matrix_g(df_mixed)

    arr = np.reshape(build_matrix_c(G), (15, 15))
    arr_mixed = np.reshape(build_matrix_c(G_mixed), (15, 15))
    np.savetxt('c_matrix.txt', arr, fmt='%.2f')
    np.savetxt('c_mixed_matrix.txt', arr_mixed, fmt='%.2f')
    show_heat_map(arr)
    show_heat_map(arr_mixed)
