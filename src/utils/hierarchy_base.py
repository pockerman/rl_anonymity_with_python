"""
HierarchyBase. A hierarchy represents as series of transformations
that can be applied on data. For example assume that the
data field has the value 'foo' a hierarchy of transformations then may be
the following list ['fo*', 'f**', '***']. If this hierarchy is fully applied
on 'foo' then 'foo' will be completely  suppressed
"""

import abc
from typing import Any


class HierarchyBase(metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    