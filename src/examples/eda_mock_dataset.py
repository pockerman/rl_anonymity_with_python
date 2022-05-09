import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, \
    get_gender_hierarchy, get_salary_bins, load_mock_subjects
from src.examples.helpers.plot_utils import plot_categorical_data

sns.set()


def plot_categorical(data, col_name: str, x_label: str,
                          y_label: str, title: str):
    sns.countplot(data.get_column(col_name=col_name))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_numeric(data, col_name: str, x_label: str,
                          y_label: str, title: str, bins=25):
    # get the salary columns
    column = data.get_column(col_name=col_name)

    plt.hist(column, bins=bins, density=True, alpha=0.75)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

PLOT_DEFAULT = False
PLOT_K_ANONYMITY_NO_SUPRESS = False
PLOT_K_ANONYMITY_SUPRESS = False
PLOT_Q_LEARNING = False
PLOT_SARSA = True

if __name__ == '__main__':

    mock_data = load_mock_subjects()
    print("Number of rows {0}".format(mock_data.n_rows))
    print("Number of rows {0}".format(mock_data.n_columns))

    if PLOT_DEFAULT:

        mock_data = load_mock_subjects()

        # plot the salary distribution
        plot_numeric(data=mock_data, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution no distortion.",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=mock_data, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts no distortion.")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=mock_data, col_name="ethnicity",
                              title="Ethnicity column counts no distortion", x_label="Ethinicity",
                              y_label="Frequency", data_new_vals=data_new_vals)

    if PLOT_K_ANONYMITY_NO_SUPRESS:
        # load kanonimity_data no suppression

        kanonymity_data_no_suppress = load_mock_subjects(
            filename="/home/alex/qi3/drl_anonymity/src/examples/k_anonymity_data/kanonymity_distorted_0.csv",
            normalized_columns=[],
            column_types={"NHSno": str,
                      "given_name": str,
                      "surname": str,
                      "gender": str,
                      "dob": str,
                      "ethnicity": str,
                      "education": str,
                      "salary": float,
                      "mutation_status": str,
                      "preventative_treatment": str,
                      "diagnosis": int},
            change_column_vals={"diagnosis": [('N', 0)], "salary":[("*", 0.0)]})

        # plot the salary distribution
        plot_numeric(data=kanonymity_data_no_suppress, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution K-anonymity (no suppress option).",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=kanonymity_data_no_suppress, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts K-anonymity (no suppress option).")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=kanonymity_data_no_suppress, col_name="ethnicity",
                              title="Ethnicity column counts K-anonymity (no suppress option)",
                              x_label="Ethinicity",
                              y_label="Frequency", data_new_vals=data_new_vals)

    if PLOT_K_ANONYMITY_SUPRESS:
        # load kanonimity_data no suppression

        kanonymity_data_no_suppress = load_mock_subjects(
            filename="/home/alex/qi3/drl_anonymity/src/examples/k_anonymity_data/kanonymity_distorted_2.csv",
            normalized_columns=[],
            column_types={"NHSno": str,
                      "given_name": str,
                      "surname": str,
                      "gender": str,
                      "dob": str,
                      "ethnicity": str,
                      "education": str,
                      "salary": float,
                      "mutation_status": str,
                      "preventative_treatment": str,
                      "diagnosis": int},
            change_column_vals={"diagnosis": [('N', 0)], "salary":[("*", 0.0)]})

        # plot the salary distribution
        plot_numeric(data=kanonymity_data_no_suppress, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution K-anonymity (2% suppress option).",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=kanonymity_data_no_suppress, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts K-anonymity (2% suppress option).")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=kanonymity_data_no_suppress, col_name="ethnicity",
                              title="Ethnicity column counts K-anonymity (2% suppress option).",
                              x_label="Ethinicity",
                              y_label="Frequency", data_new_vals=data_new_vals)


    if PLOT_Q_LEARNING:

        # load the distorted dataset
        qlearn_data = load_mock_subjects(
            filename="/home/alex/qi3/drl_anonymity/src/examples/q_learning_all_cols_results/distorted_set_-2",
            normalized_columns=[],
            column_types={"NHSno": str,
                          "given_name": str,
                          "surname": str,
                          "gender": str,
                          "dob": str,
                          "ethnicity": str,
                          "education": str,
                          "salary": float,
                          "mutation_status": str,
                          "preventative_treatment": str,
                          "diagnosis": int},
            change_column_vals={"diagnosis": [('N', 0)], "salary": [("*", 0.0)]})

        # plot the salary distribution
        plot_numeric(data=qlearn_data, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution Q-learning algorithm.",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=qlearn_data, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts Q-learning algorithm.")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=qlearn_data, col_name="ethnicity",
                              title="Ethnicity column counts Q-learning algorithm.",
                              x_label="Ethnicity",
                              y_label="Frequency", data_new_vals=data_new_vals)

