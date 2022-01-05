"""
Various utilities to calculate the distance
between two datasets. All distance metrics work
accumulative
"""

from typing import TypeVar
import numpy as np

DataSet = TypeVar("DataSet")


def lp_distance(ds1: DataSet, ds2: DataSet, p=None):
    """
    Compute the Lp norms between the respective columns in the given data sets.
    This means that the two datasets should have the same schema. It is
    up to the application to ensure that the calculation is meaningless
    :param ds1: Dataset 1
    :param ds2: Dataset 2
    :param p:  The order of the norm to calculate
    :return: The calculated Lp-norm
    """

    assert ds1.schema == ds2.schema, "Invalid schema for datasets"

    distances = {}
    cols = ds1.get_columns_names()
    for col in cols:

        val1 = ds1.get_column(col_name=col)
        val2 = ds2.get_column(col_name=col)
        distances[col] = np.linalg.norm(val1 - val2, ord=p)

    return distances, sum(distances.values())
