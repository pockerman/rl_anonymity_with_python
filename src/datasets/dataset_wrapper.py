from pathlib import Path
import abc
from typing import Generic, TypeVar
import pandas as pd
import numpy as np

from src.preprocessor.cleanup_utils import read_csv, replace, change_column_types
from src.exceptions.exceptions import InvalidDataTypeException

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

    def show_head(self, n: int) -> None:
        print(self.ds.head(n))

    def describe(self):
        print(self.ds.describe())

    def info(self):
        print(self.ds.info())

    def save_to_csv(self, filename: Path, save_index: bool) -> None:
        """
        Save the underlying dataset in a csv format
        :param filename:
        :return:
        """
        self.ds.to_csv(filename, index=save_index)

    def read(self, filename: Path, **options) -> None:
        """
        Load a data set from a file
        :param filename:
        :param options:
        :return:
        """
        self.ds = read_csv(filename=filename,
                           features_drop_names=options["features_drop_names"],
                           names=options["names"])

        if "change_col_vals" in options and \
                options["change_col_vals"] is not None and \
                len(options["change_col_vals"]) != 0:
            self.ds = replace(ds=self.ds, options=options["change_col_vals"])

        # try to cast to the data types

        # get a subset of the columns to change the types
        col_names = self.get_columns_names()
        col_types = {}
        for name in col_names:
            if name in self.columns:
                col_types[name] = self.columns[name]

        self.ds = change_column_types(ds=self.ds, column_types=col_types) #self.columns)

        if "column_normalization" in options and \
                options["column_normalization"] is not None:
            for col in options["column_normalization"]:
                self.normalize_column(column_name=col)

    def normalize_column(self, column_name) -> None:
        """
        Normalizes the column with the given name using the following
        transformation:

        z_i = \frac{x_i - min(x)}{max(x) - min(x)}

        if the column is not of numeric type then this function
        throws an InvalidDataTypeException
        :param column_name:
        :return:
        """

        data_type = self.columns[column_name]

        if data_type is not type(1) and data_type is not type(1.0):
            raise InvalidDataTypeException(param_name=column_name, param_type=data_type, param_types="[int, float]")

        col_vals = self.get_column(col_name=column_name).values

        min_val = np.min(col_vals)
        max_val = np.max(col_vals)

        for i in range(len(col_vals)):
            col_vals[i] = float((col_vals[i] - min_val)) / float((max_val - min_val))

        self.ds[column_name] = col_vals

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

    def get_column(self, col_name: str):
        """
        Returns the column with the given name
        :param col_name:
        :return:
        """
        return self.ds.loc[:, col_name]

    def get_column_unique_values(self, col_name: str):
        """
       Returns the unique values for the column
       :param col_name:
       :return:
       """
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
        column = transform.act(**{"data": column.values})
        self.ds[transform.column_name] = column
