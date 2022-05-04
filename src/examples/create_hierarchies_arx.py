"""
This example shows how to create hierarchies suitable to
be loaded in the ARX tool
"""
import csv
import numpy as np
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData


def get_ethnicity_hierarchy():

    ethnicity_hierarchy = {}

    ethnicity_hierarchy["Mixed White/Asian"] = ["White/Asian", "Mixed"]
    ethnicity_hierarchy["Chinese"] = ["Asian", "Asian"]
    ethnicity_hierarchy["Indian"] = ["Asian", "Asian"]
    ethnicity_hierarchy["Mixed White/Black African"] = ["White/Black", "Mixed"]
    ethnicity_hierarchy["Black African"] = ["African", "Black"]
    ethnicity_hierarchy["Asian other"] = ["Asian", "Asian"]
    ethnicity_hierarchy["Black other"] = ["Black", "Other"]
    ethnicity_hierarchy["Mixed White/Black Caribbean"] = ["White/Black", "Mixed"]
    ethnicity_hierarchy["Mixed other"] = ["Mixed", "Mixed"]
    ethnicity_hierarchy["Arab"] = ["Asian", "Asian"]
    ethnicity_hierarchy["White Irish"] = ["Irish", "European"]
    ethnicity_hierarchy["Not stated"] = ["Not stated", "Not stated"]
    ethnicity_hierarchy["White Gypsy/Traveller"] = ["White", "White"]
    ethnicity_hierarchy["White British"] = ["British", "European"]
    ethnicity_hierarchy["Bangladeshi"] = ["Asian", "Asian"]
    ethnicity_hierarchy["White other"] = ["White", "White"]
    ethnicity_hierarchy["Black Caribbean"] = ["Caribbean", "Black"]
    ethnicity_hierarchy["Pakistani"] = ["Asian", "Asian"]

    return ethnicity_hierarchy


if __name__ == '__main__':

    mock_data = MockSubjectsData(FEATURES_DROP_NAMES=[],
                                 NORMALIZED_COLUMNS=["salary"],
                                 COLUMNS_TYPES={"gender": str, "ethnicity": str, "education": int,
                                                "salary": float, "diagnosis": int, "preventative_treatment": str,
                                                 "mutation_status": int, "NHSno": int, "given_name": str, "surname": str,
                                                         "dob": str})

    ds = MockSubjectsLoader(mock_data)

    """
    # specify the columns to drop
    drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment",
                                                             "education", "mutation_status"]
    MockSubjectsLoader.FEATURES_DROP_NAMES = drop_columns

    # do a salary normalization
    MockSubjectsLoader.NORMALIZED_COLUMNS = ["salary"]

    # specify the columns to use
    MockSubjectsLoader.COLUMNS_TYPES = {"ethnicity": str, "salary": float, "diagnosis": int,
                                        "gender": str}
    ds = MockSubjectsLoader()
    """

    """
    ehnicity_map = get_ethnicity_hierarchy()
    # get the ethincity column loop over
    # the values and create the hierarchy file
    filename = "/home/alex/qi3/drl_anonymity/data/hierarchies/ethnicity_hierarchy.csv"
    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=";")

        ethnicity_column = ds.get_column(col_name="ethnicity").values

        for val in ethnicity_column:

            if val not in ehnicity_map:
                raise ValueError("Value {0} not in ethnicity map")

            row = [val]
            row.extend(ehnicity_map[val])
            writer.writerow(row)

    # get the salary column
    filename = "/home/alex/qi3/drl_anonymity/data/hierarchies/salary_hierarchy.csv"

    # create bins for the salary generalization
    unique_salary = ds.get_column_unique_values(col_name="salary")
    unique_salary.sort()

    # modify slightly the max value because
    # we get out of bounds for the maximum salary
    bins = np.linspace(unique_salary[0], unique_salary[-1] + 1, 10)

    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=";")

        start = bins[0]
        for i in range(1, bins.shape[0]): #ethnicity_column:
            end = bins[i]

            row = [start, end]
            writer.writerow(row)
            start = end

    """
    # get the salary column
    filename = "/home/alex/qi3/drl_anonymity/data/hierarchies/gender_hierarchy.csv"

    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=";")

        gender_column = ds.get_column(col_name="gender").values

        for val in gender_column:

            row = [val, '*']
            writer.writerow(row)

