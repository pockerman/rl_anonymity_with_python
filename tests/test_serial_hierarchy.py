"""
Unit tests for SerialHierarchy
"""
import unittest
from src.utils.serial_hierarchy import SerialHierarchy


class TestSerialHierarchy(unittest.TestCase):

    def test_constructor(self):

        hierarchy = SerialHierarchy(values={"val1": "val2", "val2": "val3"})
        self.assertEqual(2, len(hierarchy))

    def test_add(self):
        hierarchy = SerialHierarchy(values={})
        hierarchy.add(key="key1", values="val1")

        self.assertEqual("val1", hierarchy["key1"])


if __name__ == '__main__':
    unittest.main()
