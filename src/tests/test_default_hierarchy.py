import unittest
from src.utils.serial_hierarchy import SerialHierarchy


class TestDefaultHierarchy(unittest.TestCase):

    def test_iteration(self):
        values = ["test", "tes*", "te**", "t***", "****"]
        d_hierarchy = SerialHierarchy(values=values)

        self.assertEqual(d_hierarchy.value, values[0], "Invalid hierarchy value")
        next(d_hierarchy.__iter__())
        self.assertEqual(d_hierarchy.value, values[1], "Invalid hierarchy value")


if __name__ == '__main__':
    unittest.main()
