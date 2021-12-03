from pathlib import Path
import abc
from typing import Generic, TypeVar
import pandas as pd
import numpy as np

from preprocessor.cleanup_utils import read_csv, replace, change_column_types

DS = TypeVar("DS")
HierarchyBase = TypeVar('HierarchyBase')


class DSWrapper(Generic[DS], metaclass=abc.ABCMeta):

    def __init__(self) -> None:
        super(DSWrapper, self).__init__()
        self.ds = None

    @abc.abstractmethod
    def read(self, filename: Path, **options) -> None:
        """
        Load a data set from a file
        :param filename:
        :param options:
        :return: None
        """


class PandasDSWrapper(DSWrapper[pd.DataFrame]):

    """
    Simple wrapper to a pandas DataFrame object.
    Facilitates various actions on the original dataset
    """

    def __init__(self, columns: dir) -> None:
        super(PandasDSWrapper, self).__init__()

        self.columns: dir = columns

        # map that holds the hierarchy to be applied
        # on each column in the dataset
        self.column_hierarchy = {}

    def n_rows(self) -> int:
        """
        Returns the number of rows of the data set
        :return:
        """

        return self.ds.shape[0]

    def n_columns(self) -> int:
        """
        Returns the number of rows of the data set
        :return:
        """
        return self.ds.shape[1]

    def read(self, filename: Path,  **options) -> None:
        """
        Load a data set from a file
        :param filename:
        :param options:
        :return:
        """
        self.ds = read_csv(filename=filename,
                           features_drop_names=options["features_drop_names"],
                           names=options["names"])

        if "change_col_vals" in options:
            self.ds = replace(ds=self.ds, options=options["change_col_vals"])

        # try to cast to the data types
        self.ds = change_column_types(ds=self.ds, column_types=self.columns) 

    def set_columns_to_type(self, col_name_types) -> None:
        self.ds.astype(dtype=col_name_types)

    def attach_column_hierarchy(self, col_name: str, hierarchy: HierarchyBase):
        self.column_hierarchy[col_name] = hierarchy

    def get_column(self, col_name: str):
        return self.ds.loc[:, col_name]

    def get_column_unique_values(self, col_name: str):
        # what are the unique values?

        col = self.get_column(col_name=col_name)
        vals = col.values.ravel()

        return pd.unique(vals)

    def get_columns_types(self):
        return list(self.ds.dtypes)

    def get_column_type(self, col_name: str):
        return self.ds[col_name].dtype

    def get_columns_names(self):
        return list(self.ds.columns)

    def sample_column(self):

        col_names = self.get_columns_names()
        col_idx = np.random.choice(col_names, 1)
        return self.get_column(col_name=col_names[col_idx])

