from pathlib import Path

import pandas as pd

from src.datasets.datasets_loaders import MockSubjectsLoader
from src.datasets.dataset_wrapper import PandasDSWrapper
from src.preprocessor.cleanup_utils import replace, change_column_types
from src.utils.numeric_distance_type import NumericDistanceType
from src.utils.string_distance_calculator import StringDistanceType
from src.utils.distortion_calculator import DistortionCalculationType, DistortionCalculator

def load_q_learn():
    # specify the columns to drop
    drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment", "gender",
                                                             "education", "mutation_status"]
    MockSubjectsLoader.FEATURES_DROP_NAMES = []  # drop_columns

    # do a salary normalization so that we work with
    # salaries in [0, 1] this is needed as we will
    # be using normalized distances
    MockSubjectsLoader.NORMALIZED_COLUMNS = []

    # specify the columns to use
    MockSubjectsLoader.COLUMNS_TYPES = {"ethnicity": str, "salary": float, "diagnosis": int}
    MockSubjectsLoader.NAMES = ["ethnicity", "salary", "diagnosis"]
    MockSubjectsLoader.FILENAME = "/home/alex/qi3/drl_anonymity/src/examples/q_learn_distorted_sets/distorted_set_-2"
    ds_q_learn = MockSubjectsLoader()

    assert ds_q_learn.n_columns == 3, "Invalid number of columns {0} not equal to 3".format(ds_q_learn.n_columns)
    return ds_q_learn


def load_k_anonymity():

    ds_k_anonymity = PandasDSWrapper(columns={})

    options = {"features_drop_names": ["NHSno", "given_name", "surname", "dob",
                                       "preventative_treatment", "gender",
                                       "education", "mutation_status"
                                       ],
               "names": ["NHSno", "given_name", "surname", "gender",
                         "dob", "ethnicity", "education", "salary",
                         "mutation_status", "preventative_treatment", "diagnosis"]}

    ds_k_anonymity.read(filename=Path("/home/alex/qi3/drl_anonymity/src/examples/q_learn_distorted_sets/kanonymity_distorted.csv"),
                        **options)

    assert ds_k_anonymity.n_columns == 3, "Invalid number of columns {0} not equal to 3".format(ds_k_anonymity.n_columns)
    return ds_k_anonymity


def load_original_dataset():

    ds_original = PandasDSWrapper(columns={"gender": str, "ethnicity": str, "education": int,
                       "salary": float, "diagnosis": int, "preventative_treatment": str,
                       "mutation_status": int, })

    options_org = {"features_drop_names": ["NHSno", "given_name", "surname", "dob",
                                       "preventative_treatment", "gender",
                                       "education", "mutation_status"],
               "names": ["NHSno", "given_name", "surname", "gender",
                         "dob", "ethnicity", "education", "salary",
                         "mutation_status", "preventative_treatment", "diagnosis"],
               "change_col_vals": {"diagnosis": [('N', 0)]},
               "column_normalization": ["salary"]}

    ds_original.read(filename=Path("../../data/mocksubjects.csv"), **options_org)

    assert ds_original.n_columns == 3, "Invalid number of columns {0} not equal to 3".format(
        ds_original.n_columns)
    return ds_original


if __name__ == '__main__':

    ds_q_learn = load_q_learn()
    ds_k_anonymity = load_k_anonymity()

    print("Show head for Q learning dataset")
    #ds_q_learn.show_head(10)
    ds_q_learn.describe()
    ds_q_learn.info()

    print("Show head for Kanonymity learning dataset")
    #ds_k_anonymity.show_head(10)
    ds_k_anonymity.describe()
    ds_k_anonymity.info()

    # update the values * in salary to zero
    # and then update the datatype
    options = {"salary": [('*', 0.0)]}
    ds_k_anonymity.ds = replace(ds=ds_k_anonymity.ds, options=options)
    columns = {"salary": float}
    ds_k_anonymity.ds = change_column_types(ds=ds_k_anonymity.ds, column_types=columns)

    # verify
    ds_k_anonymity.info()

    # we need to preprocess the K anonymity
    # dataset so that we can compare


    n_new_rows = ds_q_learn.n_rows - ds_k_anonymity.n_rows
    rows = []
    for i in range(n_new_rows):
        ds_k_anonymity.ds.loc[-1] = ['*', 0.0, 1]
        ds_k_anonymity.ds.index += 1
        #rows.append(pd.Series())

    ds_k_anonymity.info()
    #ds_k_anonymity.show_head(10)


    ds_org = load_original_dataset()
    ds_org.info()
    ds_org.show_head(20)

    # compute distortions
    distortion_calculator = DistortionCalculator(
        numeric_column_distortion_metric_type=NumericDistanceType.L2_AVG,
        string_column_distortion_metric_type=StringDistanceType.COSINE_NORMALIZE,
        dataset_distortion_type=DistortionCalculationType.SUM)

    ethnicity_org = ds_org.get_column("ethnicity").values
    ethnicity_org = "".join(ethnicity_org)

    ethnicity_q_dist = ds_q_learn.get_column("ethnicity").values
    ethnicity_q_dist = "".join(ethnicity_q_dist)

    ethnicity_kanonymity_dist = ds_k_anonymity.get_column("ethnicity").values
    ethnicity_kanonymity_dist = "".join(ethnicity_kanonymity_dist)

    ethnicity_qlearn_dist = distortion_calculator.calculate(ethnicity_org, ethnicity_q_dist, datatype='str')
    ethnicity_kanonymity_dist = distortion_calculator.calculate(ethnicity_org, ethnicity_kanonymity_dist, datatype='str')

    salary_org = ds_org.get_column("salary").values
    salary_q_dist = ds_q_learn.get_column("salary").values
    salary_kanonymity_dist = ds_k_anonymity.get_column("salary").values

    salary_q_dist = distortion_calculator.calculate(salary_org, salary_q_dist, datatype='float')
    salary_kanonymity_dist = distortion_calculator.calculate(salary_org, salary_kanonymity_dist, datatype='float')

    print("============================================================")
    print("Column:     {0}            {1}".format("ethnicity", "Salary"))
    print("QLearn:     {0}        {1}".format(ethnicity_qlearn_dist, salary_q_dist))
    print("KAnonymity: {0}        {1}".format(ethnicity_kanonymity_dist, salary_kanonymity_dist))
    print("Total dataset distortion (QLearn) {0}".format( salary_q_dist + ethnicity_qlearn_dist))
    print("Total dataset distortion (Kanonymity) {0}".format(ethnicity_kanonymity_dist + salary_kanonymity_dist))
    print("============================================================")
