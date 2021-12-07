from pathlib import Path
from datasets.dataset_wrapper import PandasDSWrapper


class MockSubjectsLoader(PandasDSWrapper):

    DEFAULT_COLUMNS = {"gender": str, "ethnicity": str, "education": int,
                       "salary": int, "diagnosis": int, "preventative_treatment": str,
                       "mutation_status": int, }

    FILENAME = Path("../data/mocksubjects.csv")

    FEATURES_DROP_NAMES = ["NHSno", "given_name", "surname", "dob"]

    NAMES = ["NHSno", "given_name", "surname", "gender",
             "dob", "ethnicity", "education", "salary",
             "mutation_status", "preventative_treatment", "diagnosis"]

    DROP_NA = True

    CHANGE_COLS_VALS = {"diagnosis": [('N', 0)]}

    def __init__(self):
        super(MockSubjectsLoader, self).__init__(columns=MockSubjectsLoader.DEFAULT_COLUMNS)
        self.read(filename=MockSubjectsLoader.FILENAME, **{"features_drop_names": MockSubjectsLoader.FEATURES_DROP_NAMES,
                                                            "names": MockSubjectsLoader.NAMES,
                                                            "drop_na": MockSubjectsLoader.DROP_NA,
                                                            "change_col_vals": MockSubjectsLoader.CHANGE_COLS_VALS})