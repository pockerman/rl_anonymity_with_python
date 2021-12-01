from pathlib import Path
import abc
from typing import Generic, TypeVar
import pandas as pd
import numpy as np

from preprocessor.cleanup_utils import read_csv

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
        :return:
        """


class PandasDSWrapper(DSWrapper[pd.DataFrame]):
    def __init__(self) -> None:
        super(PandasDSWrapper, self).__init__()

        # map that holds the hierarchy to be applied
        # on each column in the dataset
        self.column_hierarchy = {}

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
        return self.ds.dtype

    def get_column_type(self, col_name: str):
        return self.ds[col_name].dtype

    def get_columns_names(self):
        return list(self.ds.columns)

    def sample_column(self):

        col_names = self.get_columns_names()
        col_idx = np.random.choice(col_names, 1)
        return self.get_column(col_name=col_names[col_idx])

