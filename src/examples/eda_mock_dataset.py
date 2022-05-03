import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, \
    get_gender_hierarchy, get_salary_bins, load_mock_subjects
from src.examples.helpers.plot_utils import plot_categorical_data


if __name__ == '__main__':

    mock_data = load_mock_subjects()

    # get the salary columns
    salary = mock_data.get_column(col_name="salary")

    plt.hist(salary, bins=25, density=True, alpha=0.75)
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.title("Salary column distribution no distortion.")
    #plt.show()

    # salary k-anonimity
    # salary q-learning
    kanonymity_data = load_mock_subjects(
        filename="/home/alex/qi3/drl_anonymity/src/examples/q_learning_three_columns_results/kanonymity_distorted.csv",
        normalized_columns=[],
    column_types={"NHSno":str,
                  "given_name":str,
                  "surname": str,
                  "gender": str,"dob": str,
                  "ethnicity":str, "education":str,
                  "salary": float,"mutation_status": str,
                  "preventative_treatment":str,"diagnosis":int},
    change_column_vals={"diagnosis": [('N', 0)], "salary":[("*", 0.0)]})

    # get the salary columns
    salary = kanonymity_data.get_column(col_name="salary")

    plt.hist(salary, bins=25, density=True, alpha=0.75)
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.title("K-anonymity Salary column distribution")
    #plt.show()

    data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                     "Black African": "BA", "*": "*", "Asian other": "AO",
                     "White Irish": "WI", "White other": "WO",
                     "Mixed White/Asian": "M W/A",
                     "Mixed White/Black African": "M W/B A",
                     "Mixed White/Black Caribbean": "M W/B C",
                     "Mixed other": "MO", "Black other": "BO"}
    plot_categorical_data(data=kanonymity_data, col_name="ethnicity",
                          title="K-anonymity Ethnicity", x_label="Ethinicity category",
                          y_label="Frequency", data_new_vals=data_new_vals)

    # salary q-learning
    qlearning_data = load_mock_subjects(
        filename="/home/alex/qi3/drl_anonymity/src/examples/q_learning_all_cols_results/distorted_set_1000",
        normalized_columns=[])

    # get the salary columns
    salary = qlearning_data.get_column(col_name="salary")

    plt.hist(salary, bins=25, density=True, alpha=0.75)
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.title("Qlearning Salary column distribution")
    #plt.show()


    # salary a2c distorted
    a2c_data = load_mock_subjects(filename="/home/alex/qi3/drl_anonymity/src/examples/a2c_all_cols_multi_state_results/distorted_set_14",
                                  normalized_columns=[])

    # get the salary columns
    salary = a2c_data.get_column(col_name="salary")

    plt.hist(salary, bins=25, density=True, alpha=0.75)
    plt.xlabel("Salary")
    plt.ylabel("Frequency")
    plt.title("A2C Salary column distribution")
    #plt.show()
