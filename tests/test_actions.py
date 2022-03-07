import unittest
import pytest
import numpy as np

from src.spaces.actions import ActionSuppress, ActionStringGeneralize, ActionIdentity, ActionType
from src.spaces.actions import ActionNumericBinGeneralize, ActionNumericStepGeneralize


class TestActions(unittest.TestCase):

    def test_identity_action_type(self):
        action = ActionIdentity(column_name="col1")

        self.assertEqual("col1", action.column_name)
        self.assertEqual(ActionType.IDENTITY, action.action_type)

    def test_step_generalization_action_type(self):
        action = ActionNumericStepGeneralize(column_name="col1", step=1.0)

        self.assertEqual("col1", action.column_name)
        self.assertEqual(ActionType.GENERALIZE, action.action_type)

    def test_step_generalization_action_act(self):
        action = ActionNumericStepGeneralize(column_name="col1", step=1.0)

        data = [1.0, 2.0, 3.0]
        new_data = action.act(**{"data": data})
        self.assertEqual([2.0, 4.0, 6.0], new_data)

    def test_suppress_action_type(self):
        action = ActionSuppress(column_name="col1", suppress_table=None)

        self.assertEqual("col1", action.column_name)
        self.assertEqual(ActionType.SUPPRESS, action.action_type)

    def test_string_generalization_action_type(self):

        action = ActionStringGeneralize(column_name="col1", generalization_table=None)

        self.assertEqual("col1", action.column_name)
        self.assertEqual(ActionType.GENERALIZE, action.action_type)

    def test_string_generalization_action_act(self):

        table = {"col1": "Alex", "col2": "Alex2", "col3": "Alex3",
                 "Alex": "1", "Alex2": "1", "Alex3": "1", "1": "1"}
        data = ["col1", "col2", "col3"]
        action = ActionStringGeneralize(column_name="col1", generalization_table=table)

        update_data = action.act(**{"data": data})

        self.assertEqual(["Alex", "Alex2", "Alex3"], update_data)

        # act again on the data
        update_data = action.act(**{"data": update_data})
        self.assertEqual(["1", "1", "1"], update_data)

    def test_bin_generalization_action_type(self):
        action = ActionNumericBinGeneralize(column_name="col", generalization_table= [10])
        self.assertEqual("col", action.column_name)
        self.assertEqual(ActionType.GENERALIZE, action.action_type)

    def test_bin_generalization_action_bins(self):

        values = [i + 100*i for i in range(10)]
        bins = np.linspace(values[0], values[-1], 10)
        action = ActionNumericBinGeneralize(column_name="col", generalization_table=bins)

        bins = action.bins

        self.assertEqual(9, len(bins))

        self.assertEqual(0.0, bins[0][0])
        self.assertEqual(101., bins[0][1])
        self.assertEqual(101.0, bins[1][0])
        self.assertEqual(202.0, bins[1][1])
        self.assertEqual(202.0, bins[2][0])
        self.assertEqual(303.0, bins[2][1])
        self.assertEqual(303.0, bins[3][0])
        self.assertEqual(404.0, bins[3][1])
        self.assertEqual(404.0, bins[4][0])
        self.assertEqual(505.0, bins[4][1])

        self.assertEqual(505.0, bins[5][0])
        self.assertEqual(606.0, bins[5][1])

        self.assertEqual(606.0, bins[6][0])
        self.assertEqual(707.0, bins[6][1])

        self.assertEqual(707.0, bins[7][0])
        self.assertEqual(808.0, bins[7][1])

        self.assertEqual(808.0, bins[8][0])
        self.assertEqual(909.0, bins[8][1])

    def test_bin_generalization_action_act_fail_low_bound(self):

        values = [i + 100*i for i in range(10)]
        bins = np.linspace(values[0], values[-1], 10)
        action = ActionNumericBinGeneralize(column_name="col", generalization_table=bins)

        with pytest.raises(ValueError) as execinfo:
            action.act(**{"data": [-1]})

    def test_bin_generalization_action_act_fail_upper_bound(self):

        values = [i + 100*i for i in range(10)]
        bins = np.linspace(values[0], values[-1], 10)
        action = ActionNumericBinGeneralize(column_name="col", generalization_table=bins)

        with pytest.raises(ValueError) as execinfo:
            action.act(**{"data": [values[-1]]})

    def test_bin_generalization_action_act(self):

        values = [i + 100*i for i in range(10)]
        bins = np.linspace(values[0], values[-1], 10)
        action = ActionNumericBinGeneralize(column_name="col", generalization_table=bins)

        vals = [values[i] for i in range(1, len(values) -1 )]
        vals = action.act(**{"data": vals})

        for i, val in enumerate(vals):

            bin_idx = np.digitize(val, action.table)
            bin = action.bins[bin_idx - 1]
            self.assertEqual(0.5*(bin[0] + bin[1]), val)


if __name__ == '__main__':
    unittest.main()
