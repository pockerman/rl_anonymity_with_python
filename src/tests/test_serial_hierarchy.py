"""
Unit tests for SerialHierarchy
"""
import unittest
from src.utils.serial_hierarchy import SerialHierarchy


class TestSerialHierarchy(unittest.TestCase):


    def test_add_values(self):

        hierarchy = SerialHierarchy()
        hierarchy.add("val1", "val2", "val3")
        self.assertEqual(1, len(hierarchy))
        self.assertEqual(list, type(hierarchy[0]))


if __name__ == '__main__':
    unittest.main()
