import csv
import pandas as pd
from pathlib import Path
from typing import List


def read_csv(filename: Path, features_drop_names: List[str], names: List[str], delimiter=','):
    """
    Read the csv file specified at the given filename
    :param filename: The path to the filw to read
    :param features_drop_names: Features to drop
    :param names: Names of columns
    :return:
    """

    df = pd.read_csv(filepath_or_buffer=filename, sep=delimiter, header=0, names=names)

    if len(features_drop_names) != 0:
        df.drop(features_drop_names, inplace=True, axis=1)

    # drop all rows with NaN
    df.dropna(inplace=True, axis=0)

    return df