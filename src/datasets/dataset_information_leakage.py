"""
Utilities for calculating the information leakage
for a dataset
"""
import numpy as np
from typing import TypeVar
from src.exceptions.exceptions import InvalidSchemaException
from src.datasets.dataset_distances import lp_distance

DataSet = TypeVar("DataSet")


def info_leakage(ds1: DataSet, ds2: DataSet, column_distances: dict = None, p=None) -> tuple:
    """
    Returns the information leakage between the two data sets
    :param ds1:
    :param ds2:
    :param column_dists: A dictionary that holds numeric distances to use if a column
    is of type string
    :return:
    """

    if ds1.schema != ds2.schema:
        raise InvalidSchemaException(message="Invalid schema for datasets")

    if column_distances is None:
        return lp_distance(ds1=ds1, ds2=ds2, p=p)

    distances = {}
    cols = ds1.get_columns_names()
    for col in cols:

        if col in column_distances:
            # get the total distortion of the column
            distances[col] = column_distances[col]
        else:

            val1 = ds1.get_column(col_name=col)
            val2 = ds2.get_column(col_name=col)
            distances[col] = np.linalg.norm(val1 - val2, ord=p)

    sum_distances = sum(distances.values())
    return distances, sum_distances



