"""Loads the three-columns dataset for the examples

"""
from pathlib import Path
from typing import Any
import numpy as np

from src.utils.serial_hierarchy import SerialHierarchy
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData
from src.spaces.discrete_state_environment import DiscreteStateEnvironment
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.maths.distortion_calculator import DistortionCalculationType, DistortionCalculator
from src.maths.numeric_distance_type import NumericDistanceType
from src.maths.string_distance_calculator import StringDistanceType
from src.utils.reward_manager import RewardManager
from src.spaces.env_type import DiscreteEnvType


N_LAYERS = 5
N_BINS = 10
N_EPISODES = 1000
OUTPUT_MSG_FREQUENCY = 100
GAMMA = 0.99
ALPHA = 0.1

MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.3
OUT_OF_MAX_BOUND_REWARD = -1.0
OUT_OF_MIN_BOUND_REWARD = -1.0
IN_BOUNDS_REWARD = 5.0
N_ROUNDS_BELOW_MIN_DISTORTION = 10
#SAVE_DISTORTED_SETS_DIR = "/home/alex/qi3/drl_anonymity/src/examples/semi_grad_sarsa/distorted_set"
REWARD_FACTOR = 0.95
PUNISH_FACTOR = 2.0


def get_gender_hierarchy():
    hierarchy = SerialHierarchy(values={"F": "*", "M": "*", "*": "*"})
    return hierarchy


def get_ethinicity_hierarchy():
    ethnicity_hierarchy = SerialHierarchy(values={})

    ethnicity_hierarchy["Mixed White/Asian"] = "White/Asian"
    ethnicity_hierarchy["White/Asian"] = "Mixed"

    ethnicity_hierarchy["Chinese"] = "Asian"
    ethnicity_hierarchy["Indian"] = "Asian"
    ethnicity_hierarchy["Mixed White/Black African"] = "White/Black"
    ethnicity_hierarchy["White/Black"] = "Mixed"

    ethnicity_hierarchy["Black African"] = "African"
    ethnicity_hierarchy["African"] = "Black"
    ethnicity_hierarchy["Asian other"] = "Asian"
    ethnicity_hierarchy["Black other"] = "Black"
    ethnicity_hierarchy["Mixed White/Black Caribbean"] = "White/Black"
    ethnicity_hierarchy["White/Black"] = "Mixed"

    ethnicity_hierarchy["Mixed other"] = "Mixed"
    ethnicity_hierarchy["Arab"] = "Asian"
    ethnicity_hierarchy["White Irish"] = "Irish"
    ethnicity_hierarchy["Irish"] = "European"
    ethnicity_hierarchy["Not stated"] = "Not stated"
    ethnicity_hierarchy["White Gypsy/Traveller"] = "White"
    ethnicity_hierarchy["White British"] = "British"
    ethnicity_hierarchy["British"] = "European"
    ethnicity_hierarchy["Bangladeshi"] = "Asian"
    ethnicity_hierarchy["White other"] = "White"
    ethnicity_hierarchy["Black Caribbean"] = "Caribbean"
    ethnicity_hierarchy["Caribbean"] = "Black"
    ethnicity_hierarchy["Pakistani"] = "Asian"

    ethnicity_hierarchy["European"] = "European"
    ethnicity_hierarchy["Mixed"] = "Mixed"
    ethnicity_hierarchy["Asian"] = "Asian"
    ethnicity_hierarchy["Black"] = "Black"
    ethnicity_hierarchy["White"] = "White"
    return ethnicity_hierarchy


def load_mock_subjects() -> MockSubjectsLoader:

    COLUMN_TYPES = MockSubjectsData().COLUMNS_TYPES
    COLUMN_TYPES["NHSno"] = str
    mock_data = MockSubjectsData(FILENAME=Path("../../data/mocksubjects.csv"),
                                 FEATURES_DROP_NAMES=[],
                                 NORMALIZED_COLUMNS=["salary"],
                                 COLUMNS_TYPES=COLUMN_TYPES)

    ds = MockSubjectsLoader(mock_data)

    return ds


def load_discrete_env(env_type: DiscreteEnvType, n_states: int,
                      min_distortion: Any, max_distortion: Any,
                      total_min_distortion: float, total_max_distortion: float,
                      punish_factor: float, column_types: dict,
                      action_space: ActionSpace,
                      save_distoreted_sets_dir: str,
                      use_identifying_column_dist_in_total_dist: bool,
                      use_identifying_column_dist_factor: float) -> DiscreteStateEnvironment:
        mock_ds = load_mock_subjects()

        action_space.shuffle()

        env = DiscreteStateEnvironment.from_options(data_set=mock_ds,
                                                    action_space=action_space,
                                                    distortion_calculator=DistortionCalculator(
                                                        numeric_column_distortion_metric_type=NumericDistanceType.L2_AVG,
                                                        string_column_distortion_metric_type=StringDistanceType.COSINE_NORMALIZE,
                                                        dataset_distortion_type=DistortionCalculationType.SUM),
                                                    reward_manager=RewardManager(
                                                        bounds=(total_min_distortion, total_max_distortion),
                                                        out_of_max_bound_reward=OUT_OF_MAX_BOUND_REWARD,
                                                        out_of_min_bound_reward=OUT_OF_MIN_BOUND_REWARD,
                                                        in_bounds_reward=IN_BOUNDS_REWARD,
                                                        min_distortions=min_distortion, max_distortions=max_distortion,
                                                        punish_factor=punish_factor),
                                                    gamma=GAMMA,
                                                    reward_factor=REWARD_FACTOR,
                                                    min_distortion=min_distortion,
                                                    min_total_distortion=total_min_distortion,
                                                    max_distortion=max_distortion,
                                                    max_total_distortion=total_max_distortion,
                                                    n_rounds_below_min_distortion=N_ROUNDS_BELOW_MIN_DISTORTION,
                                                    distorted_set_path=Path(save_distoreted_sets_dir),
                                                    n_states=n_states, env_type=env_type, column_types=column_types,
                                                    use_identifying_column_dist_in_total_dist=use_identifying_column_dist_in_total_dist,
                                                    use_identifying_column_dist_factor=use_identifying_column_dist_factor)

        return env