from pathlib import Path
import abc
from typing import Generic, TypeVar
import pandas as pd
import numpy as np

from src.preprocessor.cleanup_utils import read_csv, replace, change_column_types

DS = TypeVar("DS")
HierarchyBase = TypeVar('HierarchyBase')
Transform = TypeVar("Transform")


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

    @property
    def n_rows(self) -> int:
        """
        Returns the number of rows of the data set
        :return:
        """
        return self.ds.shape[0]

    @property
    def n_columns(self) -> int:
        """
        Returns the number of rows of the data set
        :return:
        """
        return self.ds.shape[1]

    @property
    def schema(self) -> dict:
        return pd.io.json.build_table_schema(self.ds)

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

    def sample_column_name(self) -> str:
        """
        Samples a name from the columns
        :return: a column name
        """
        names = self.get_columns_names()
        return np.random.choice(names)

    def set_columns_to_type(self, col_name_types) -> None:
        """
        Set the types of the columns
        :param col_name_types:
        :return:
        """
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

    def apply_column_transform(self, column_name: str, transform: Transform) -> None:
        """
        Apply the given transformation on the underlying dataset
        :param column_name: The column to transform
        :param transform: The transformation to apply
        :return: None
        """

        # get the column
        column = self.get_column(col_name=column_name)
        column = transform.act(**{"data": column})
        self.ds[transform.column_name] = column





