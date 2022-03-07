import unittest
import pytest


from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionSuppress, ActionIdentity, ActionStringGeneralize
from src.utils.serial_hierarchy import SerialHierarchy
from src.datasets.datasets_loaders import MockSubjectsLoader
from src.utils.reward_manager import RewardManager
from src.utils.string_distance_calculator import StringDistanceType
from src.datasets.dataset_information_leakage import info_leakage


class TestDatasetInfoLeakage(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup the PandasDSWrapper to be used in the tests
        :return: None
        """
        pass

        """
        # load the dataset
        self.ds = MockSubjectsLoader()

        # specify the action space. We need to establish how these actions
        # are performed
        self.action_space = ActionSpace(n=4)

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


        self.action_space.add_many(ActionSuppress(column_name="gender", suppress_table={"F": SerialHierarchy(values=['*', ]),
                                                                                        'M': SerialHierarchy(values=['*', ])}),
                                   ActionIdentity(column_name="salary"), ActionIdentity(column_name="education"),
                                   ActionStringGeneralize(column_name="ethnicity", generalization_table=self.generalization_table))
        self.reward_manager = RewardManager()
        """

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_info_leakage_1(self):
        """
        No distortion is applied on the data set so total distortion
        should be zero
        """
        pass

        """
        env_config = EnvConfig()
        env_config.action_space = self.action_space
        env_config.reward_manager = self.reward_manager
        env_config.data_set = self.ds
        env_config.start_column = "gender"
        env_config.gamma = 0.99

        # create the environment
        env = Environment(env_config=env_config)

        # initialize text distances
        env.initialize_text_distances(distance_type=StringDistanceType.COSINE)

        distances, sum_distances = info_leakage(ds1=env.data_set, ds2=env.start_ds, column_distances=env.column_distances)

        # no leakage should exist as no trasformation is applied
        self.assertEqual(0.0, sum_distances)
        """

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_info_leakage_2(self):
        """
        We apply distortion on column gender
        """
        pass

        """
        env_config = EnvConfig()
        env_config.action_space = self.action_space
        env_config.reward_manager = self.reward_manager
        env_config.data_set = self.ds
        env_config.start_column = "gender"
        env_config.gamma = 0.99

        # create the environment
        env = Environment(env_config=env_config)

        # initialize text distances
        env.initialize_text_distances(distance_type=StringDistanceType.COSINE)

        action = env.action_space.get_action_by_column_name(column_name="gender")

        env.step(action=action)

        distances, sum_distances = info_leakage(ds1=env.data_set, ds2=env.start_ds,
                                                column_distances=env.column_distances)

        # leakage should exist as we suppress the gender column
        self.assertNotEqual(0.0, sum_distances)
        """


if __name__ == '__main__':
    unittest.main()
