"""
This example shows how to create hierarchies suitable to
be loaded in the ARX tool
"""
import csv
from src.datasets.datasets_loaders import MockSubjectsLoader


def get_ethnicity_hierarchy():

    ethnicity_hierarchy = {}

    ethnicity_hierarchy["Mixed White/Asian"] = ["White/Asian", "Mixed"]
    ethnicity_hierarchy["Chinese"] = ["Asian", "Asian"]
    ethnicity_hierarchy["Indian"] = ["Asian", "Asian"]
    ethnicity_hierarchy["Mixed White/Black African"] = ["White/Black", "Mixed"]
    ethnicity_hierarchy["Black African"] = ["Black", "African"]
    ethnicity_hierarchy["Asian other"] = ["Asian", "Other"]
    ethnicity_hierarchy["Black other"] = ["Black", "Other"]
    ethnicity_hierarchy["Mixed White/Black Caribbean"] = ["White/Black", "Mixed"]
    ethnicity_hierarchy["Mixed other"] = ["Mixed", "Mixe"]
    ethnicity_hierarchy["Arab"] = ["Asian", "Asian"]
    ethnicity_hierarchy["White Irish"] = ["Irish", "European"]
    ethnicity_hierarchy["Not stated"] = ["Not stated", "Not stated"]
    ethnicity_hierarchy["White Gypsy/Traveller"] = ["White", "White"]
    ethnicity_hierarchy["White British"] = ["British", "European"]
    ethnicity_hierarchy["Bangladeshi"] = ["Asian", "Asian"]
    ethnicity_hierarchy["White other"] = ["White", "White"]
    ethnicity_hierarchy["Black Caribbean"] = ["Black", "Caribbean"]
    ethnicity_hierarchy["Pakistani"] = ["Asian", "Asian"]

    return ethnicity_hierarchy


if __name__ == '__main__':

    # specify the columns to drop
    drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment", "gender",
                                                             "education", "mutation_status"]
    MockSubjectsLoader.FEATURES_DROP_NAMES = drop_columns

    # do a salary normalization
    MockSubjectsLoader.NORMALIZED_COLUMNS = ["salary"]

    # specify the columns to use
    MockSubjectsLoader.COLUMNS_TYPES = {"ethnicity": str, "salary": float, "diagnosis": int}
    ds = MockSubjectsLoader()

    ehnicity_map = get_ethnicity_hierarchy()
    # get the ethincity column loop over
    # the values and create the hierarchy file
    filename = "/home/alex/qi3/drl_anonymity/data/hierarchies/ethnicity_hierarchy.csv"
    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter=",")

        ethnicity_column = ds.get_column(col_name="ethnicity").values

        for val in ethnicity_column:

            if val not in ehnicity_map:
                raise ValueError("Value {0} not in ethnicity map")

            row = [val]
            row.extend(ehnicity_map[val])
            writer.writerow(row)

