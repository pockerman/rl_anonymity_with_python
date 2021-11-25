import unittest
from pathlib import Path

from preprocessor.cleanup_utils import read_csv


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


if __name__ == '__main__':
    unittest.main()