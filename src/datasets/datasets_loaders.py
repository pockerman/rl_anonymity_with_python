"""
Utility class that allows to load the mocksubjects.csv
dataset and perform various transformations and cleaning
on it
"""

from pathlib import Path
from typing import List
from dataclasses import dataclass, field

from src.datasets.dataset_wrapper import PandasDSWrapper


@dataclass(init=True, repr=True)
class MockSubjectsData(object):

    # Path to the dataset file
    FILENAME: Path = Path("/home/alex/qi3/drl_anonymity/data/mocksubjects.csv") #("../../data/mocksubjects.csv")

    # the assumed column types. We use this map to cast
    # the types of the columns
    COLUMNS_TYPES: dict = field(default_factory=lambda: {"gender": str, "ethnicity": str, "education": int,
                     "salary": int, "diagnosis": int, "preventative_treatment": str,
                     "mutation_status": int,})

    # features to drop
    FEATURES_DROP_NAMES: List[str] = field(default_factory=lambda: ["NHSno", "given_name", "surname", "dob"])

    # Names of the columns in the dataset
    NAMES: List[str] = field(default_factory=lambda: ["NHSno", "given_name", "surname", "gender",
                                              "dob", "ethnicity", "education", "salary",
                                              "mutation_status", "preventative_treatment", "diagnosis"])

    # option to drop NaN
    DROP_NA: bool = True

    # Map that holds for each column the transformations
    # we want to apply for each value
    CHANGE_COLS_VALS: dict = field(default_factory=lambda: {"diagnosis": [('N', 0)]})

    # list of columns to be normalized
    NORMALIZED_COLUMNS: List[str] = field(default_factory=list)


class MockSubjectsLoader(PandasDSWrapper):
    """The class MockSubjectsLoader. Loads the  mocksubjects.csv
    """

    @classmethod
    def from_options(cls, *, filename: Path,
                     column_types: dir, features_drop_names: List[str],
                     names: List[str], drop_na: bool, change_col_vals: dict, column_normalization: List[str]):

        data = MockSubjectsData(FILENAME=filename, COLUMNS_TYPES=column_types,
                                FEATURES_DROP_NAMES=features_drop_names, NAMES=names,
                                DROP_NA=drop_na, CHANGE_COLS_VALS=change_col_vals,
                                NORMALIZED_COLUMNS=column_normalization)
        return cls(data=data)

    def __init__(self, data: MockSubjectsData, do_read: bool = True):
        super(MockSubjectsLoader, self).__init__(columns=data.COLUMNS_TYPES)

        if do_read:
            self.read(filename=data.FILENAME,
                      **{"features_drop_names": data.FEATURES_DROP_NAMES,
                         "names": data.NAMES,
                         "drop_na": data.DROP_NA,
                         "change_col_vals": data.CHANGE_COLS_VALS,
                         "column_normalization": data.NORMALIZED_COLUMNS})
