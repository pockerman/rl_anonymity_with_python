"""
Utility class that allows to load the mocksubjects.csv
dataset and perform various transformations and cleaning
on it
"""

from pathlib import Path
from src.datasets.dataset_wrapper import PandasDSWrapper


class MockSubjectsLoader(PandasDSWrapper):
    """
    The class MockSubjectsLoader. Loads the  mocksubjects.csv
    """

    # Path to the dataset file
    FILENAME = Path("../../data/mocksubjects.csv")

    # the assumed column types. We use this map to cast
    # the types of the columns
    COLUMNS_TYPES = {"gender": str, "ethnicity": str, "education": int,
                       "salary": int, "diagnosis": int, "preventative_treatment": str,
                       "mutation_status": int, }

    # features to drop
    FEATURES_DROP_NAMES = ["NHSno", "given_name", "surname", "dob"]

    # Names of the columns in the dataset
    NAMES = ["NHSno", "given_name", "surname", "gender",
             "dob", "ethnicity", "education", "salary",
             "mutation_status", "preventative_treatment", "diagnosis"]

    # option to drop NaN
    DROP_NA = True

    # Map that holds for each column the transformations
    # we want to apply for each value
    CHANGE_COLS_VALS = {"diagnosis": [('N', 0)]}

    # list of columns to be normalized
    NORMALIZED_COLUMNS = []

    def __init__(self):
        super(MockSubjectsLoader, self).__init__(columns=MockSubjectsLoader.COLUMNS_TYPES)
        self.read(filename=MockSubjectsLoader.FILENAME,
                  **{"features_drop_names": MockSubjectsLoader.FEATURES_DROP_NAMES,
                     "names": MockSubjectsLoader.NAMES,
                     "drop_na": MockSubjectsLoader.DROP_NA,
                     "change_col_vals": MockSubjectsLoader.CHANGE_COLS_VALS,
                     "column_normalization": MockSubjectsLoader.NORMALIZED_COLUMNS})
