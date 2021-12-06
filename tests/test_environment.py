import unittest
from pathlib import Path

import pytest

from spaces.environment import Environment
from spaces.action_space import ActionSpace
from exceptions.exceptions import Error
from utils.string_sequence_calculator import DistanceType
from datasets.dataset_wrapper import PandasDSWrapper


class TestEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the PandasDSWrapper to be used in the tests
        :return: None
        """

        # read the data
        filename = Path("../data/mocksubjects.csv")

        cols_types = {"gender": str, "ethnicity": str, "education": int,
                       "salary": int, "diagnosis": int, "preventative_treatment": str,
                       "mutation_status": int, }

        self.ds = PandasDSWrapper(columns=cols_types)
        self.ds.read(filename=filename, **{"features_drop_names": ["NHSno", "given_name", "surname", "dob"],
                                            "names": ["NHSno", "given_name", "surname", "gender",
                                                "dob", "ethnicity", "education", "salary",
                                                "mutation_status", "preventative_treatment", "diagnosis"],
                                           "drop_na": True,
                                           "change_col_vals": {"diagnosis": [('N', 0)]}})

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_prepare_column_states_throw_Error(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        with pytest.raises(Error):
            env.prepare_column_states()

    #@pytest.mark.skip(reason="no way of currently testing this")
    def test_prepare_column_states(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        env.initialize_text_distances(distance_type=DistanceType.COSINE)
        env.prepare_column_states()

    def test_get_numeric_ds(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        env.initialize_text_distances(distance_type=DistanceType.COSINE)
        env.prepare_column_states()

        tensor = env.get_ds_as_tensor()

        # test the shape of the tensor
        shape0 = tensor.size(dim=0)
        shape1 = tensor.size(dim=1)

        self.assertEqual(shape0, env.start_ds.n_rows())
        self.assertEqual(shape1, env.start_ds.n_columns())





if __name__ == '__main__':
    unittest.main()