import unittest
import pytest
import numpy as np
from pathlib import Path
import pandas as pd
from preprocessor.cleanup_utils import read_csv, replace, change_column_types


class TestPreprocessor(unittest.TestCase):

    def test_read_csv(self):

        filename = Path("../data/mocksubjects.csv")
        features_drop_names = ["NHSno", "given_name", "surname"]
        names = ["NHSno", "given_name", "surname", "gender",
                 "dob", "ethnicity", "education", "salary",
                 "mutation_status", "preventative_treatment", "diagnosis"]
        df = read_csv(filename=filename, features_drop_names=features_drop_names,
                      names=names)

        print(df.head(n=10))

    def test_replace(self):

        df = pd.DataFrame({"col1": ["val1", "val2", "val3", "val4"], "col2": [0, 1, 2, 3]})

        options = {"col1": [("val1", "5"), ("val2", "10")], "col2": [(0, 20), (3, 5)]}
        replace(df, options=options)

        self.assertEqual(df["col1"].values[0], "5")
        self.assertEqual(df["col1"].values[1], "10")
        self.assertEqual(df["col2"].values[0], 20)
        self.assertEqual(df["col2"].values[3], 5)

    def test_change_column_types(self):

        df = pd.DataFrame({"col1": ["1", "2", "3", "4"]})

        options = {"col1": int}
        df = change_column_types(df, column_types=options)

        self.assertEqual(df["col1"].dtypes, np.int64)


if __name__ == '__main__':
    unittest.main()