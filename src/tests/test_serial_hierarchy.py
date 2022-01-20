"""
Unit tests for SerialHierarchy
"""
import unittest
from src.utils.serial_hierarchy import SerialHierarchy


class TestSerialHierarchy(unittest.TestCase):

    def test_constructor(self):

        hierarchy = SerialHierarchy(values={"val1": "val2", "val2": "val3"})
        self.assertEqual(2, len(hierarchy))


if __name__ == '__main__':
    unittest.main()
