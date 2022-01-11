import unittest

import unittest
from pathlib import Path

import pytest

from src.spaces.environment import Environment
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionSuppress, ActionGeneralize
from src.exceptions.exceptions import Error
from src.utils.serial_hierarchy import SerialHierarchy
from src.utils.string_distance_calculator import DistanceType
from src.datasets.dataset_wrapper import PandasDSWrapper
from src.spaces.state_space import StateSpace, State

class TestStateSpace(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the PandasDSWrapper to be used in the tests
        :return: None
        """

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

    def test_creation(self):

        action_space = ActionSpace(n=3)

        generalization_table = {"Mixed White/Asian": SerialHierarchy(values=["Mixed", ]),
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

        action_space.add(ActionGeneralize(column_name="ethnicity", generalization_table=generalization_table))

        # create the environment from the given dataset
        env = Environment(data_set=self.ds, action_space=action_space, gamma=0.99, start_column="gender")

        state_space = StateSpace()
        state_space.init_from_environment(env=env)

        print(state_space.states.keys())

        self.assertEqual(env.n_features, state_space.n)


if __name__ == '__main__':
    unittest.main()