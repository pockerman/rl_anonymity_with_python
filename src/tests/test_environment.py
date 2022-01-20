import unittest
from pathlib import Path

import pytest

from src.spaces.discrete_state_environment import Environment
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionSuppress, ActionStringGeneralize
from src.exceptions.exceptions import Error
from src.utils.serial_hierarchy import SerialHierarchy
from src.utils.string_distance_calculator import StringDistanceType
from src.datasets.dataset_wrapper import PandasDSWrapper
from src.utils.reward_manager import RewardManager


class TestEnvironment(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the PandasDSWrapper to be used in the tests
        :return: None
        """

        # specify the reward manager to use
        self.reward_manager = RewardManager()

        # read the data
        filename = Path("../../data/mocksubjects.csv")

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

        self.generalization_table = {"Mixed White/Asian": SerialHierarchy(values=["Mixed", ]),
                                "Chinese": SerialHierarchy(values=["Asian", ]),
                                "Indian": SerialHierarchy(values=["Asian", ]),
                                "Mixed White/Black African": SerialHierarchy(values=["Mixed", ]),
                                "Black African": SerialHierarchy(values=["Black", ]),
                                "Asian other": SerialHierarchy(values=["Asian", ]),
                                "Black other": SerialHierarchy(values=["Black", ]),
                                "Mixed White/Black Caribbean": SerialHierarchy(values=["Mixed", ]),
                                "Mixed other": SerialHierarchy(values=["Mixed", ]),
                                "Arab": SerialHierarchy(values=["Asian", ]),
                                "White Irish": SerialHierarchy(values=["White", ]),
                                "Not stated": SerialHierarchy(values=["Not stated"]),
                                "White Gypsy/Traveller": SerialHierarchy(values=["White", ]),
                                "White British": SerialHierarchy(values=["White", ]),
                                "Bangladeshi": SerialHierarchy(values=["Asian", ]),
                                "White other": SerialHierarchy(values=["White", ]),
                                "Black Caribbean": SerialHierarchy(values=["Black", ]),
                                "Pakistani": SerialHierarchy(values=["Asian", ])}

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_prepare_column_states_throw_Error(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        with pytest.raises(Error):
            env.prepare_columns_state()

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_prepare_column_states(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        env.initialize_text_distances(distance_type=StringDistanceType.COSINE)
        env.prepare_columns_state()

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_get_numeric_ds(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99,
                          start_column="gender", reward_manager=self.reward_manager)

        env.initialize_text_distances(distance_type=StringDistanceType.COSINE)
        env.prepare_columns_state()

        tensor = env.get_ds_as_tensor()

        # test the shape of the tensor
        shape0 = tensor.size(dim=0)
        shape1 = tensor.size(dim=1)

        self.assertEqual(shape0, env.start_ds.n_rows)
        self.assertEqual(shape1, env.start_ds.n_columns)

    def test_apply_action(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)

        action_space.add(ActionStringGeneralize(column_name="ethnicity", generalization_table=self.generalization_table))

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space,
                          gamma=0.99, start_column="gender", reward_manager=self.reward_manager)

        # this will update the environment
        env.apply_action(action=action_space[0])

        # test that the ethnicity column has been changed
        # get the unique values for the ethnicity column
        unique_col_vals = env.data_set.get_column_unique_values(col_name="ethnicity")

        unique_vals = ["Mixed", "Asian", "Not stated", "White", "Black"]
        self.assertEqual(len(unique_vals), len(unique_col_vals))
        self.assertEqual(unique_vals, unique_col_vals)

    def test_step(self):
        # specify the action space. We need to establish how these actions
        # are performed
        action_space = ActionSpace(n=1)
        action_space.add(ActionStringGeneralize(column_name="ethnicity", generalization_table=self.generalization_table))

        # create the environment and
        env = Environment(data_set=self.ds, action_space=action_space,
                          gamma=0.99, start_column="gender", reward_manager=self.reward_manager)

        action = env.sample_action()

        # this will update the environment
        time_step = env.step(action=action)


if __name__ == '__main__':
    unittest.main()