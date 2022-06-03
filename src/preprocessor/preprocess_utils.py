"""module preprocess_utils. Specifies utilities for
preprocessing a data set.

"""
import csv
import pandas as pd
from pathlib import Path
from typing import List


def read_csv(filename: Path, features_drop_names: List[str], names: List[str], delimiter=',') -> pd.DataFrame:

    """Read the csv file specified at the given filename

    Parameters
    ----------
    filename: Filename to read
    features_drop_names: Which columns to drop
    names: Column names
    delimiter: file delimiter

    Returns
    -------

    A pandas DataFrame
    """


    df = pd.read_csv(filepath_or_buffer=filename, sep=delimiter, header=0, names=names)

    if len(features_drop_names) != 0:
        df.drop(features_drop_names, inplace=True, axis=1)

    # drop all rows with NaN
    df.dropna(inplace=True, axis=0)

    return df


def replace(ds: pd.DataFrame, options: dict) -> pd.DataFrame:
    """Replace the values in the given data set according to the passed
    options. The options should specify for each column the values
    to be changed and the corresponding values to set

    Parameters
    ----------
    ds
    options

    Returns
    -------

    """

    for col in options:

        # get the values to change for each column
        vals_to_change = options[col]
        column = ds.loc[:, col]

        for old_val, new_val in vals_to_change:
            column = column.replace(to_replace=old_val, value=new_val)

        # finally update the ds
        ds[col] = column

    return ds


def change_column_types(ds, column_types) -> pd.DataFrame:
    """Change the column type

    Parameters
    ----------
    ds
    column_types

    Returns
    -------

    """
    ds = ds.astype(dtype=column_types)
    return ds
