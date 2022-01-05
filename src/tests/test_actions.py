import unittest

from src.utils.serial_hierarchy import SerialHierarchy
from src.spaces.actions import ActionSuppress


class TestActions(unittest.TestCase):

    def test_suppress_action_creation(self):

        suppress_table = {"test": SerialHierarchy(values=["test", "tes*", "te**", "t***", "****"]),
                          "do_not_test": SerialHierarchy(values=["do_not_test", "do_not_tes*", "do_not_te**", "do_not_t***", "do_not_****"])}

        suppress_action = ActionSuppress(column_name="none", suppress_table=suppress_table)

        self.assertEqual(len(suppress_action.table), 2, "Invalid table size")

    def test_suppress_action_act(self):

        data = ["test", "do_not_test", "invalid"]

        suppress_table = {"test": SerialHierarchy(values=["test", "tes*", "te**", "t***", "****"]),
                          "do_not_test": SerialHierarchy(values=["do_not_test", "do_not_tes*",
                                                                  "do_not_te**", "do_not_t***", "do_not_****"])}

        suppress_action = ActionSuppress(column_name="none", suppress_table=suppress_table)

        suppress_action.act(**{"data": data})

        self.assertEqual(data[0], "test", "Invalid suppression")
        self.assertEqual(data[1], "do_not_test", "Invalid suppression")

        suppress_action.act(**{"data": data})

        self.assertEqual(data[0], "tes*", "Invalid suppression")
        self.assertEqual(data[1], "do_not_tes*", "Invalid suppression")


if __name__ == '__main__':
    unittest.main()
