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

def plot_categorical_data(data, col_name:str, x_label: str,
                          y_label: str, title: str, data_new_vals={}):
    import seaborn as sns
    sns.set()

    ax1 = plt.subplot(111)
    if len(data_new_vals) != 0:
        data_vals = data.get_column(col_name=col_name).values

        column = data.get_column(col_name=col_name)
        col_vals = column.values

        for i in range(len(col_vals)):

            if data_vals[i] in data_new_vals:
                data_vals[i] = data_new_vals[data_vals[i]]

        data.ds[col_name] = data_vals
        sns.countplot(data.ds[col_name])
    else:
        sns.countplot(data.get_column(col_name=col_name), color='gray')


    #col_data = data.get_column(col_name=col_name)
    #vals = {}

    #for item in col_data
    #plt.bar(names, values)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if len(data_new_vals) != 0:
        legend_vals = [data_new_vals[name] + "->" + name for name in data_new_vals]
        legend_vals = ["White British->WB", "Black Caribbean->BC", "Asian other->AO", "Black African->BA"]
                     #"Black African->BA", "Asian other->AO",]

                     #"White Irish": "WI", "White other": "WO",
                     #"Mixed White/Asian": "M W/A",
                     #"Mixed White/Black African": "M W/B A",
                     #"Mixed White/Black Caribbean": "M W/B C",
                     #"Mixed other": "MO", "Black other": "BO"]
        plt.legend(labels=legend_vals, loc='upper right')

    plt.show()
