import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def show_heat_map(C):
    C = np.flipud(C)
    sns.heatmap(C, yticklabels=[i for i in range(len(C[0]), 0, -1)], xticklabels=[i for i in range(1, len(C[0]) + 1)])
    plt.show()


def build_corr_matrix(arr):
    size = len(arr)
    corr_matrix = np.empty((size, size))

    for i in range(size):
        for j in range(size):
            corr, _ = pearsonr(arr[i], arr[j])
            corr_matrix[i][j] = "%.3f" % corr
    return corr_matrix


def build_corr_by_day_range(arr, day_range):
    size = len(arr[0])
    arr_of_avg = []
    for i in range(0, size, day_range):
        corr_list = []
        for j in range(len(arr)):
            for k in range(len(arr)):
                corr, _ = pearsonr(arr[j][i:i + day_range], arr[k][i:i + day_range])
                corr_list.append(corr)
        arr_of_avg.append(np.average(corr_list))
    days = [i + 1 for i in range(0, size, day_range)]

    plt.xticks(np.arange(1, size, day_range), days[::day_range])
    plt.plot(arr_of_avg)
    # plt.show()
    return plt


def build_3d_graph_6(arr, C_dist):
    z = [h.get_height() for h in C_dist.patches]
    y = list(np.linspace(1, len(arr[0]), len(z)))
    x = list(np.linspace(0, 1, len(z)))
    X, Y = np.meshgrid(x, y)
    Z, _ = np.meshgrid(z, z)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", edgecolor="none")
    # surf = ax.plot_trisurf(x, y, z, cmap="viridis")
    fig.colorbar(surf)


def build_graph_8(C, q):
    def get_prm_lambda(l):
        return (q/2*np.pi)*(np.sqrt((lambda_plus-l)*(l - lambda_minus))/l)

    eig = np.linalg.eigvals(C)
    lambda_plus = 1 + (1/q) + 2*np.sqrt(1/q)
    lambda_minus = 1 + (1/q) - 2*np.sqrt(1/q)
    print(eig)
    prm = [get_prm_lambda(l) for l in eig]
    print(prm)
    sns.distplot(eig)
    sns.distplot(prm)
    plt.show()


if __name__ == "__main__":
    file_path = f"data{os.path.sep}data_lab_5_without_date.csv"
    df = pd.read_csv(file_path)
    df = df.transpose()
    arr = df.values

    arr_mixed = np.random.uniform(low=0, high=450, size=arr.shape)

    # Build Cross correlation matrices
    C = build_corr_matrix(arr)
    C_mixed = build_corr_matrix(arr_mixed)

    # Heat maps
    # show_heat_map(C)
    # show_heat_map(C_mixed)
    #
    # # Show distribution plots with graph 6 -> normal matrix first and then the mixed one
    #
    # # dist plot + 3d for usual matrix
    # C_dist = sns.distplot(C)
    # plt.show()
    # build_3d_graph_6(arr, C_dist)
    # plt.show()
    #
    # # dist plot + 3d for mixed matrix
    # C_mixed_dist = sns.distplot(C_mixed)
    # plt.show()
    # build_3d_graph_6(arr_mixed, C_mixed_dist)
    # plt.show()
    q = len(arr[0]) / len(arr)
    build_graph_8(C, q)
