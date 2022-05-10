import random
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, \
    get_gender_hierarchy, get_salary_bins, load_mock_subjects
from src.examples.helpers.plot_utils import plot_categorical_data
from src.maths.numeric_distance_type import NumericDistanceType
from src.maths.string_distance_calculator import StringDistanceType
from src.maths.distortion_calculator import DistortionCalculationType, DistortionCalculator

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
PLOT_K_ANONYMITY_NO_SUPRESS = True
PLOT_K_ANONYMITY_SUPRESS = False
PLOT_Q_LEARNING = True
PLOT_SARSA = True
PLOT_A2C = True

if __name__ == '__main__':

    mock_data = load_mock_subjects()
    print("Number of rows {0}".format(mock_data.n_rows))
    print("Number of rows {0}".format(mock_data.n_columns))

    # get the columns
    salary_org = mock_data.get_column(col_name="salary")
    ethnicity_org = mock_data.get_column(col_name="ethnicity")
    gender_org = mock_data.get_column(col_name="gender")

    salary_org_values = salary_org.values

    ethnicity_org_values = ethnicity_org.values
    ethnicity_org_values = "".join(ethnicity_org_values)

    gender_org_values = gender_org.values
    gender_org_values = "".join(gender_org_values)

    # compute distortions
    distortion_calculator = DistortionCalculator(
        numeric_column_distortion_metric_type=NumericDistanceType.L2_AVG,
        string_column_distortion_metric_type=StringDistanceType.COSINE_NORMALIZE,
        dataset_distortion_type=DistortionCalculationType.SUM)

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

        salary_k_ano = kanonymity_data_no_suppress.get_column(col_name="salary")
        salary_k_ano_values = salary_k_ano.values

        ethnicity_ano = kanonymity_data_no_suppress.get_column(col_name="ethnicity")
        ethnicity_ano_values = ethnicity_ano.values
        ethnicity_ano_values = "".join(ethnicity_ano_values)

        gender_ano = kanonymity_data_no_suppress.get_column(col_name="gender")
        gender_ano_values = gender_ano.values
        gender_ano_values = "".join(gender_ano_values)

        ethnicity_kanonymity_dist = distortion_calculator.calculate(ethnicity_ano_values,
                                                                    ethnicity_org_values,
                                                                    datatype='str')

        salary_kanonymity_dist = distortion_calculator.calculate(salary_k_ano_values,
                                                                 salary_org_values,
                                                                 datatype='float')

        gender_kanonymity_dist = distortion_calculator.calculate(gender_ano_values,
                                                                 gender_org_values,
                                                                 datatype='str')

        print("============================================================")
        print("Column:     |  {0}   |  {1}  | {2}".format("ethnicity", "Salary", "Gender"))
        print("KAnonymity: |  {0}   |  {1}  | {2}".format(ethnicity_kanonymity_dist, salary_kanonymity_dist, gender_kanonymity_dist))
        print("Total dataset distortion (Kanonymity) {0}".format(ethnicity_kanonymity_dist + salary_kanonymity_dist + gender_kanonymity_dist))
        print("============================================================")

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

        salary_k_ano = qlearn_data.get_column(col_name="salary")
        salary_k_ano_values = salary_k_ano.values

        ethnicity_ano = qlearn_data.get_column(col_name="ethnicity")
        ethnicity_ano_values = ethnicity_ano.values
        ethnicity_ano_values = "".join(ethnicity_ano_values)

        gender_ano = qlearn_data.get_column(col_name="gender")
        gender_ano_values = gender_ano.values
        gender_ano_values = "".join(gender_ano_values)

        ethnicity_qlearn_dist = distortion_calculator.calculate(ethnicity_ano_values,
                                                                    ethnicity_org_values,
                                                                    datatype='str')

        salary_qlearn_dist = distortion_calculator.calculate(salary_k_ano_values,
                                                                 salary_org_values,
                                                                 datatype='float')

        gender_qlearn_dist = distortion_calculator.calculate(gender_ano_values,
                                                                 gender_org_values,
                                                                 datatype='str')

        print("============================================================")
        print("Column:     |  {0}   |  {1}  | {2}".format("ethnicity", "Salary", "Gender"))
        print("Qlearn: |  {0}   |  {1}  | {2}".format(ethnicity_qlearn_dist, salary_qlearn_dist,
                                                          gender_qlearn_dist))
        print("Total dataset distortion (Kanonymity) {0}".format(
            ethnicity_qlearn_dist + salary_qlearn_dist + gender_qlearn_dist))
        print("============================================================")

    if PLOT_SARSA:

        # load the distorted dataset
        sarsa_data = load_mock_subjects(
            filename="/home/alex/qi3/drl_anonymity/src/examples/semi_grad_sarsa_all_columns/distorted_set_-2",
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
        plot_numeric(data=sarsa_data, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution semi-gradient SARSA algorithm.",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=sarsa_data, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts semi-gradient SARSA algorithm.")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=sarsa_data, col_name="ethnicity",
                              title="Ethnicity column counts semi-gradient SARSA algorithm.",
                              x_label="Ethnicity",
                              y_label="Frequency", data_new_vals=data_new_vals)

        salary_k_ano = sarsa_data.get_column(col_name="salary")
        salary_k_ano_values = salary_k_ano.values

        ethnicity_ano = sarsa_data.get_column(col_name="ethnicity")
        ethnicity_ano_values = ethnicity_ano.values
        ethnicity_ano_values = "".join(ethnicity_ano_values)

        gender_ano = sarsa_data.get_column(col_name="gender")
        gender_ano_values = gender_ano.values
        gender_ano_values = "".join(gender_ano_values)

        ethnicity_sarsa_dist = distortion_calculator.calculate(ethnicity_ano_values,
                                                                ethnicity_org_values,
                                                                datatype='str')

        salary_sarsa_dist = distortion_calculator.calculate(salary_k_ano_values,
                                                             salary_org_values,
                                                             datatype='float')

        gender_sarsa_dist = distortion_calculator.calculate(gender_ano_values,
                                                             gender_org_values,
                                                             datatype='str')

        print("============================================================")
        print("Column:     |  {0}   |  {1}  | {2}".format("ethnicity", "Salary", "Gender"))
        print("SARSA: |  {0}   |  {1}  | {2}".format(ethnicity_sarsa_dist, salary_sarsa_dist,
                                                      gender_sarsa_dist))
        print("Total dataset distortion (SARSA) {0}".format(
            ethnicity_sarsa_dist + salary_sarsa_dist + gender_sarsa_dist))
        print("============================================================")

    if PLOT_A2C:
        # load the distorted dataset
        a2c_data = load_mock_subjects(
            filename="/home/alex/qi3/drl_anonymity/src/examples/a2c_all_cols_multi_state_results/distorted_set_14",
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
        plot_numeric(data=a2c_data, col_name="salary",
                     x_label="Salary", y_label="Frequency",
                     title="Salary column distribution A2C algorithm.",
                     bins=25)

        # plot the gender distribution
        plot_categorical(data=a2c_data, col_name="gender",
                         x_label="Gender", y_label="Frequency",
                         title="Gender column counts A2C algorithm.")

        data_new_vals = {"White British": "WB", "Black Caribbean": "BC",
                         "Black African": "BA", "*": "*", "Asian other": "AO",
                         "White Irish": "WI", "White other": "WO",
                         "Mixed White/Asian": "MW/A",
                         "Mixed White/Black African": "MW/BA",
                         "Mixed White/Black Caribbean": "MW/BC",
                         "Mixed other": "MO", "Black other": "BO",
                         "White Gypsy/Traveller": "WGT"}
        plot_categorical_data(data=a2c_data, col_name="ethnicity",
                              title="Ethnicity column counts A2C algorithm.",
                              x_label="Ethnicity",
                              y_label="Frequency", data_new_vals=data_new_vals)

        salary_k_ano = a2c_data.get_column(col_name="salary")
        salary_k_ano_values = salary_k_ano.values

        ethnicity_ano = a2c_data.get_column(col_name="ethnicity")
        ethnicity_ano_values = ethnicity_ano.values
        ethnicity_ano_values = "".join(ethnicity_ano_values)

        gender_ano = a2c_data.get_column(col_name="gender")
        gender_ano_values = gender_ano.values
        gender_ano_values = "".join(gender_ano_values)

        ethnicity_a2c_dist = distortion_calculator.calculate(ethnicity_ano_values,
                                                               ethnicity_org_values,
                                                               datatype='str')

        salary_a2c_dist = distortion_calculator.calculate(salary_k_ano_values,
                                                            salary_org_values,
                                                            datatype='float')

        gender_a2c_dist = distortion_calculator.calculate(gender_ano_values,
                                                            gender_org_values,
                                                            datatype='str')

        print("============================================================")
        print("Column:     |  {0}   |  {1}  | {2}".format("ethnicity", "Salary", "Gender"))
        print("A2C: |  {0}   |  {1}  | {2}".format(ethnicity_a2c_dist, salary_a2c_dist,
                                                     gender_a2c_dist))
        print("Total dataset distortion (A2C) {0}".format(
            ethnicity_a2c_dist + salary_a2c_dist + gender_a2c_dist))
        print("============================================================")


