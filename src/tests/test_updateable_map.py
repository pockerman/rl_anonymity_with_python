import unittest
import pytest

from src.utils.updateable_map import  UpdateableMap


class TestUpdateableMap(unittest.TestCase):

    def setUp(self) -> None:

        self.map = UpdateableMap(list_size=3)
        values = ["val1", "val2", "val3", "val10"]
        key = "key1"
        self.map.insert(key=key, values=values)

        values = ["val4", "val5", "val6", "val10"]
        key = "key2"
        self.map.insert(key=key, values=values)

    def test_invalid_list_size(self):
        map = UpdateableMap(list_size=2)
        values = ["val1", "val2", "val3"]
        key = "key"

        with pytest.raises(ValueError) as e_info:
            # this should throw
            map.insert(key=key, values=values)

    def test_get_value(self):

        value = self.map.get_value("key1")
        self.assertEqual("val1", value)

        # the key should be updated
        value = self.map.get_value(key=value)
        self.assertEqual("val2", value)

        keys = list(self.map.keys())
        self.assertEqual(["val2"], keys)

        values = self.map.values()
        self.assertEqual(["key1", "val1", "val3"], list(values)[0])

        print(self.map.current_pos)










if __name__ == '__main__':
    unittest.main()