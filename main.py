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


def build_3d_graph_6(arr, day_range):
    # z = [h.get_height() for h in C_dist.patches]
    z = []
    arr = np.array(arr)

    for i in range(1, len(arr[0]), day_range):


        # C_dist = sns.distplot(build_corr_matrix(arr[0:len(arr)][i:i+day_range]))
        C_dist = sns.distplot(build_corr_matrix(arr[:, i:i + day_range]))

        z_part = [h.get_height() for h in C_dist.patches]
        z.extend(z_part)
    y = list(np.linspace(1, len(arr[0]), len(z)))
    x = list(np.linspace(0, 1, len(z)))
    X, Y = np.meshgrid(x, y)
    Z, _ = np.meshgrid(z, z)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap="plasma", edgecolor="none")
    # surf = ax.plot_trisurf(x, y, z, cmap="viridis")
    fig.colorbar(surf)
    return surf


def build_graph_8(arr):
    def get_prm_lambda(l, q):
        up_left = lambda_plus - l
        up_right = l - lambda_minus
        up_ = (up_left * up_right)
        up = np.sqrt(up_)
        down = l
        left = (q / 2 * np.pi)
        right = (up / down)
        # print(f"UP_LEFT: {up_left}")
        # print(f"UP_RIGHT: {up_right}")
        # print(f"UP_: {up_}")
        # print(f"UP: {up}")
        # print(f"DOWN: {down}")
        # print(f"LEFT: {left}")
        # print(f"RIGHT: {right}")
        return left * right

    eig = np.linalg.eigvals(C)
    L = len(arr[0])
    N = len(arr)
    R_matrix = np.dot(arr, np.transpose(arr)) / L
    R_eig = np.linalg.eigvals(R_matrix)
    # print(R_matrix)
    print(["%.3f" % r for r in R_eig])
    Q = L / N
    # print(Q)
    lambda_plus = 1 + (1 / Q) + 2 * np.sqrt(1 / Q)
    lambda_minus = 1 + (1 / Q) - 2 * np.sqrt(1 / Q)
    print(lambda_plus)
    print(lambda_minus)
    # prm = [get_prm_lambda(l, Q) for l in R_matrix if lambda_minus <= l <= lambda_plus]
    prm = [get_prm_lambda(l, Q) for l in R_eig]
    print(prm)
    sns.distplot(eig)
    sns.distplot(prm)
    plt.legend(["eigs", "prm"])
    plt.show()


def build_graph_10(eig):
    return sum([x ** 4 for x in eig])


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
    show_heat_map(C)
    show_heat_map(C_mixed)
    #
    # # Show distribution plots with graph 6 -> normal matrix first and then the mixed one
    #
    # # dist plot + 3d for usual matrix
    C_dist = sns.distplot(C)
    plt.show()
    s1 = build_3d_graph_6(arr, 5)
    plt.show()
    #
    # # dist plot + 3d for mixed matrix
    C_mixed_dist = sns.distplot(C_mixed, color="orange")
    plt.show()
    s2 = build_3d_graph_6(arr_mixed, 5)
    plt.show()

    C_dist = sns.distplot(C)
    C_mixed_dist = sns.distplot(C_mixed)
    plt.legend(["вихідна матриця", "перемішана матриця"])
    plt.show()
    build_corr_by_day_range(arr, 5)
    build_corr_by_day_range(arr_mixed, 5)
    plt.legend(["вихідна матриця", "перемішана матриця"])
    plt.show()
