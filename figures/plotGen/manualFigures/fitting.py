# -*- coding: utf-8 -*-

"""
@author: alexl
"""

import numpy as np
import pandas as pd
from rawdata import table_dict6Li, table_dict3He
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib import rcParams
from copy import deepcopy as copy

rcParams["text.usetex"] = True
rcParams["font.family"] = "serif"

# The values of omegaH and Ntotmax present in the dataset, note that
# np.arange does not include the endpoints 14 and 26

thetas = np.array(
    [40, 55, 75, 90, 110, 125, 145, 159]
)  # The angles present in the dataset
boundset = {}
# Basically forces the result to be in the range [lowerVal, upperVal]
# with boundset[theta] corrispodning to the theta value
# I checked the values for 40 and 159 carefully, the rest may need to be redone
boundset[40] = [237, 247]
boundset[55] = [187, 200]
boundset[75] = [142, 151]
boundset[90] = [127, 136]
boundset[110] = [134, 142]
boundset[125] = [151, 168]
boundset[145] = [182, 195]
boundset[159] = [195, 225]


table_dict = table_dict6Li


def main():
    def func(x, b, c, d):
        """
        Define a function passed to plotMe that will be fitted at the angles
        you specify in thetaPlot
        if you change the number of parameters this function takes
        you will also have to change "bounds" to be the correct length
        """
        x = x - 3
        return b + c * x**-2.0 + d * x**-3.0
        # return b + (c / x) + d * x * (np.e**-x)

    # plot6Li(func)
    # plot3He(func)
    plot6LiSqueeze()


def plot3He(funcfit):
    omegaHs = np.arange(12, 26, 2)
    ntos = np.arange(6, 24, 2)
    energyAngles = [(50, 30), (120, 150)]

    def getCS(omegaH, Ntotmax, energy, angle):
        key = (omegaH, Ntotmax, energy, angle)
        return table_dict3He.get(key, None)

    pointSize = 20
    xs = np.arange(np.min(ntos), np.max(ntos), 0.01)
    markers = ["o", "^", "v", "x", "+", "$M$", "*", "D", "s", "H", "h"]
    colors = 3 * ["b", "g", "r", "c", "m", "k"]

    # fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    _, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs = axs.flatten("F")
    theta = 30
    for axs_i, ax in enumerate(axs):
        energy, theta = energyAngles[axs_i]
        for i, omeg in enumerate(omegaHs):
            ntostmp = []
            vals = []
            for nto in ntos:
                cs = getCS(omeg, nto, energy, theta)
                if cs is not None:
                    vals.append(cs)
                    ntostmp.append(nto)

            yData = np.array([getCS(omeg, nto, energy, theta) for nto in ntos])
            out, covar = curve_fit(
                funcfit, ntos, yData, maxfev=10000000
            )  # "out" is an array

            ax.plot(xs, funcfit(xs, *out), color=colors[i])

            yDatafit = funcfit(ntos, *out)  # automatically unpacks "out"

            numer = np.sum((yDatafit - yData) ** 2)
            denom = np.sum((yData - np.mean(yData)) ** 2)
            residue = 1 - (numer / denom)

            label = "omegaH=" + str(omegaHs[i])
            label += f"\\;$R^2={np.round(residue,5)}$"

            ax.scatter(
                np.array(ntostmp),
                np.array(vals),
                s=pointSize,
                marker=markers[i],
                color=colors[i],
                label=label,
            )

            ax.legend()
            titleStr = f"$\\omega={energy},\\;\\theta={theta}$"
            ax.set_title(titleStr)

    plt.tight_layout()
    plt.show()


def plot6LiSqueeze():
    omegaHs = np.arange(14, 26, 2)
    ntos = np.arange(6, 16, 2)
    table_dict = table_dict6Li

    def getCS(omegaH, ntotmax, theta):
        return table_dict.get((omegaH, ntotmax, theta), None)

    pointSize = 12
    markers = ["o", "^", "v", "x", "+", "$M$"]
    colors = ["b", "g", "r", "c", "m", "k"]
    _, axs = plt.subplots(3, 3, figsize=(18, 15))
    axs = axs.flatten("F")
    ax1 = axs[0]
    # Adjust spacing between subplots
    # plt.subplots_adjust(hspace=0.5)

    # Plotting on the first subplot (ax1)
    for theta in thetas:
        for i, omeg in enumerate(omegaHs):
            ntostmp = []
            vals = []
            for nto in ntos:
                cs = getCS(omeg, nto, theta)
                if cs is not None:
                    vals.append(cs)
                    ntostmp.append(nto)

            if theta == thetas[0]:
                ax1.scatter(
                    theta + np.array(ntostmp),
                    np.array(vals),
                    s=pointSize,
                    marker=markers[i],
                    color=colors[i],
                    label="omegaH=" + str(omegaHs[i]),
                )
            else:
                ax1.scatter(
                    theta + np.array(ntostmp),
                    np.array(vals),
                    s=pointSize,
                    marker=markers[i],
                    color=colors[i],
                )

    # Set titles and labels for the first subplot
    titleStr = r"Compton ${}^{6}\mathrm{Li}$ Cross section vs $\theta$"
    titleStr += (
        "\n"
        + "Clusters of points show Ntotmax and omegaH \n dependence of cross section for same $\\theta$"
    )
    ax1.set_title(titleStr)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\Omega$")
    ax1.legend()

    # theta = thetaPlot
    # In this loop for each value of theta passed in thetaPlot
    # We make a scatter plot, then make a fit and plot the fit over it

    # for i, theta in enumerate(thetaPlot):
    out = []
    for i, theta in enumerate(thetas):
        ax = axs[i + 1]
        maxY = -np.inf
        minY = np.inf
        resultDict = {}
        for i, omeg in enumerate(omegaHs):
            ntostmp = []
            vals = []
            for nto in ntos:
                cs = getCS(omeg, nto, theta)
                if cs is not None:
                    vals.append(cs)
                    ntostmp.append(nto)
                    if cs > maxY:
                        maxY = cs
                    if cs < minY:
                        minY = cs
            # This is the scatterplot
            resultDict[omeg] = np.array([ntostmp, vals])
            resultDict[(omeg, "color")] = markers[i], colors[i]
            ax.scatter(
                np.array(ntostmp),
                np.array(vals),
                s=pointSize * 1.7,
                marker=markers[i],
                color=colors[i],
                alpha=0.2,
                label=f"omegaH={omeg}",
            )
        omegaGood = np.array([16, 18])
        slopes = np.zeros(2)
        while True:
            for j, omega in enumerate(omegaGood):
                # print("omegaGood=", omegaGood)
                # print("theta=", theta)

                xs, ys = resultDict[omega]
                x1, x2 = xs[-1], xs[-2]
                y1, y2 = ys[-1], ys[-2]

                slopes[j] = (y1 - y2) / (x1 - x2)

            if slopes[0] < 0:
                loc = np.where(omegaHs == omegaGood[0])[0][0]
                assert loc > 0
                omegaGood[0] = omegaHs[loc - 1]

            if slopes[1] > 0:
                loc = np.where(omegaHs == omegaGood[1])[0][0]
                assert loc < len(omegaHs)
                omegaGood[1] = omegaHs[loc + 1]
            if slopes[0] > 0 and slopes[1] < 0:
                break

        xs1, y1 = resultDict[omegaGood[0]]
        xs2, y2 = resultDict[omegaGood[1]]
        y1 = y1[-1]
        y2 = y2[-1]
        sigma = abs(y1 - y2)
        x1 = xs1[-1]
        x2 = xs2[-1]
        m1, m2 = slopes
        xmeet = (x1 * m1 - x2 * m2 - y1 + y2) / (m1 - m2)
        ymeet = m1 * (xmeet - x1) + y1
        ymeet2 = m2 * (xmeet - x2) + y2
        assert abs(ymeet - ymeet2) < 0.01

        for k in range(2):
            ntos, vals = resultDict[omegaGood[k]]
            marker, color = resultDict[(omegaGood[k], "color")]
            ax.scatter(ntos, vals, marker=marker, color=color, alpha=1.0)
            xs = np.arange(np.max(ntos) - 2, xmeet + 2, 0.1)
            ax.plot(xs, m1 * (xs - x1) + y1, linestyle="--", c="r")
            ax.plot(xs, m2 * (xs - x2) + y2, linestyle="--", c="r")

        ax.scatter(xmeet, ymeet, marker="X", c="k", label="Estimated Value")
        out.append([theta, ymeet, sigma])
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\Omega$")
        legend = ax.legend()
        for lh in legend.legend_handles:
            lh.set_alpha(1)

        ax.set_title(f"$\\bar{{x}}={np.round(ymeet,4)},\\;\\sigma={np.round(sigma,4)}$")

        ax.set_xlabel("nTotMax")
        # ax.set_xlim([np.min(xs) * 0.8, np.max(xs)])

    plt.tight_layout()
    plt.show()
    fileout = array_to_table(
        np.array(out), None, ["theta", "dσ/dΩ [µbarns]", "uncertainty"]
    )
    fileout += "\nUncertainty listed here is an overestimate of uncertainty arrising from Ntotmax truncation"
    fileout += "\nOther uncertainties may contribute"
    save_string_to_file("6LiCrossSection.txt", fileout)
    print(fileout)


def array_to_table(array, row_labels, col_labels):
    """
    Converts a MxN numpy array into a table with labeled rows and columns.

    Parameters:
        array (numpy.ndarray): A 2D numpy array of shape (M, N).
        row_labels (list): A list of labels for the rows (length M).
        col_labels (list): A list of labels for the columns (length N).

    Returns:
        pandas.DataFrame: A DataFrame representing the table with the given labels.
    """
    # Check if the array is 2-dimensional.
    if array.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    # Check if the number of labels matches the dimensions of the array.
    # if len(row_labels) != array.shape[0]:
    #     raise ValueError(
    #         "The number of row labels must match the number of rows in the array"
    #     )
    if len(col_labels) != array.shape[1]:
        raise ValueError(
            "The number of column labels must match the number of columns in the array"
        )

    # Create a DataFrame with the specified row and column labels.
    df = pd.DataFrame(array, index=row_labels, columns=col_labels)

    return df.to_string(index=False)


def plot6Li(funcfit):
    omegaHs = np.arange(14, 26, 2)
    ntos = np.arange(6, 14, 2)
    table_dict = table_dict6Li

    def getCS(omegaH, ntotmax, theta):
        return table_dict.get((omegaH, ntotmax, theta), None)

    pointSize = 12
    markers = ["o", "^", "v", "x", "+", "$M$"]
    colors = ["b", "g", "r", "c", "m", "k"]
    # if isinstance(thetaPlot, (float, int)):
    #     thetaPlot = [thetaPlot]

    # Create a figure with 2 subplots (vertically stacked)
    # _, axs = plt.subplots(len(thetaPlot) + 1, 1, figsize=(xSize, xSize * 1.5))
    _, axs = plt.subplots(3, 3, figsize=(18, 15))
    axs = axs.flatten("F")
    ax1 = axs[0]
    # Adjust spacing between subplots
    # plt.subplots_adjust(hspace=0.5)

    # Plotting on the first subplot (ax1)
    for theta in thetas:
        for i, omeg in enumerate(omegaHs):
            ntostmp = []
            vals = []
            for nto in ntos:
                cs = getCS(omeg, nto, theta)
                if cs is not None:
                    vals.append(cs)
                    ntostmp.append(nto)

            if theta == thetas[0]:
                ax1.scatter(
                    theta + np.array(ntostmp),
                    np.array(vals),
                    s=pointSize,
                    marker=markers[i],
                    color=colors[i],
                    label="omegaH=" + str(omegaHs[i]),
                )
            else:
                ax1.scatter(
                    theta + np.array(ntostmp),
                    np.array(vals),
                    s=pointSize,
                    marker=markers[i],
                    color=colors[i],
                )

    # Set titles and labels for the first subplot
    titleStr = r"Compton ${}^{6}\mathrm{Li}$ Cross section vs $\theta$"
    titleStr += (
        "\n"
        + "Clusters of points show Ntotmax and omegaH \n dependence of cross section for same $\\theta$"
    )
    ax1.set_title(titleStr)
    ax1.set_xlabel(r"$\theta$")
    ax1.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\Omega$")
    ax1.legend()

    # theta = thetaPlot
    # In this loop for each value of theta passed in thetaPlot
    # We make a scatter plot, then make a fit and plot the fit over it

    # for i, theta in enumerate(thetaPlot):
    for i, theta in enumerate(thetas):
        ax = axs[i + 1]
        maxY = -np.inf
        minY = np.inf

        for i, omeg in enumerate(omegaHs):
            ntostmp = []
            vals = []
            for nto in ntos:
                cs = getCS(omeg, nto, theta)
                if cs is not None:
                    vals.append(cs)
                    ntostmp.append(nto)
                    if cs > maxY:
                        maxY = cs
                    if cs < minY:
                        minY = cs
            # This is the scatterplot
            ax.scatter(
                np.array(ntostmp),
                np.array(vals),
                s=pointSize * 1.7,
                marker=markers[i],
                color=colors[i],
            )

        # This is the fitted plot
        xs = np.arange(np.min(ntos), np.max(ntos) * 4.5, 0.01)

        bounds = np.array(
            [
                boundset[theta],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
            ]
        )
        xData = copy(ntos)
        bs = []
        infPoints = []
        for j, omeg in enumerate(omegaHs):
            yData = np.array([getCS(omeg, nto, theta) for nto in ntos])
            if None not in yData:
                out, covar = curve_fit(
                    funcfit, xData, yData, bounds=bounds.T, maxfev=10000000
                )  # "out" is an array
                bs.append(out[0])

                ys = funcfit(xs, *out)  # automatically unpacks "out"
                yPoints = funcfit(ntos, *out)  # automatically unpacks "out"

                yDatafit = funcfit(ntos, *out)  # automatically unpacks "out"
                numer = np.sum((yDatafit - yData) ** 2)
                denom = np.sum((yData - np.mean(yData)) ** 2)
                residue = 1 - (numer / denom)

                label = f"omegaH={omeg}"
                label += f"\\;$R^2={np.round(residue,5)}$"
                infPoints.append(funcfit(np.inf, *out))
                ax.plot(xs, ys, label=label, color=colors[j])

        ax.axhline(y=boundset[theta][0], linestyle="--", color="r")
        ax.axhline(
            y=boundset[theta][1],
            linestyle="--",
            color="r",
            label="Asymptotic bound of fitted curve",
        )

        ax.set_ylim([minY - 8, maxY + 8])
        ax.set_ylabel(r"$\mathrm{d}\sigma/\mathrm{d}\Omega$")
        ax.legend()

        bs = np.array(bs)
        sigma = np.std(bs)
        mean = np.mean(bs)
        ax.set_title(
            r"$\theta="
            + str(theta)
            + "$ degrees, $\\bar{{x}}="
            + str(np.round(mean, 3))
            + r"\;\;\sigma="
            + str(np.round(sigma, 3))
            + "$"
        )

        ax.set_xlabel("nTotMax")
        ax.set_xlim([np.min(xs) * 0.8, np.max(xs)])

    plt.tight_layout()
    plt.show()


def containsClose(arr, val, delta=0.01):
    diff = abs(arr - val)
    return np.min(diff) < delta


def save_string_to_file(file_path: str, content: str):
    """
    Saves the provided string content to a text file at the specified file path.

    Parameters:
        file_path (str): The path (including filename) where the text file will be saved.
        content (str): The string content to write into the file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


if __name__ == "__main__":
    main()
