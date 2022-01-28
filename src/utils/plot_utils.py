import matplotlib.pyplot as plt
import numpy as np


def plot_running_avg(avg_array, steps: int,
                     xlabel: str, ylabel: str,
                     title: str, show_grid: bool = True):
    """
    Plot a running average of the values in the arra
    :param title:
    :param xlabel:
    :param ylabel:
    :param show_grid:
    :param avg_array:
    :param steps:
    :return:
    """

    running_avg = np.empty(avg_array.shape[0])
    for t in range(avg_array.shape[0]):
        running_avg[t] = np.mean(avg_array[max(0, t-steps): (t+1)])

    plt.plot(running_avg)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if show_grid:
        plt.grid()

    plt.show()