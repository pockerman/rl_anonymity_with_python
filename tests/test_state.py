"""
Unit tests for StateSpace
"""

import unittest
import pytest

from src.spaces.state import State


class TestState(unittest.TestCase):

    def test_contains(self):
        state = State()
        state.column_names = ["col1", "col2", "col3"]
        self.assertTrue("col1" in state)

    def test_iterate(self):
        state = State()
        state.column_names = ["col1", "col2", "col3"]

        for name in state:
            self.assertTrue(name in state)


if __name__ == '__main__':
    unittest.main()
