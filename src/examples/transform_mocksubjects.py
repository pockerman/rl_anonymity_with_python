import csv
from pathlib import Path
from src.datasets.datasets_loaders import MockSubjectsLoader

if __name__ == '__main__':
    # specify the columns to drop
    drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment", "gender",
                                                             "education", "mutation_status"]
    MockSubjectsLoader.FEATURES_DROP_NAMES = []  # drop_columns

    # do a salary normalization
    MockSubjectsLoader.NORMALIZED_COLUMNS = ["salary"]

    # specify the columns to use
    MockSubjectsLoader.COLUMNS_TYPES = {"gender": str, "ethnicity": str, "education": int,
                                        "salary": float, "diagnosis": int, "preventative_treatment": str,
                                        "mutation_status": int, }
    ds = MockSubjectsLoader()

    ds.save_to_csv(filename=Path("/home/alex/qi3/drl_anonymity/data/hierarchies/normalized_salary_mocksubjects.csv"),
                   save_index=False)
