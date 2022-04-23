"""Loads the three-columns dataset for the examples

"""
from pathlib import Path
from typing import Any
import numpy as np

from src.utils.serial_hierarchy import SerialHierarchy
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData
from src.spaces.discrete_state_environment import DiscreteStateEnvironment
from src.spaces.action_space import ActionSpace
from src.maths.distortion_calculator import DistortionCalculationType, DistortionCalculator
from src.maths.numeric_distance_type import NumericDistanceType
from src.maths.string_distance_calculator import StringDistanceType
from src.utils.reward_manager import RewardManager
from src.spaces.env_type import DiscreteEnvType


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


def get_salary_bins(ds: MockSubjectsLoader, n_states: int):
    # create bins for the salary generalization
    unique_salary = ds.get_column_unique_values(col_name="salary")
    unique_salary.sort()

    # modify slightly the max value because
    # we get out of bounds for the maximum salary
    bins = np.linspace(unique_salary[0], unique_salary[-1] + 1, n_states)
    return bins


def load_discrete_env(env_type: DiscreteEnvType, n_states: int,
                      min_distortion: Any, max_distortion: Any,
                      total_min_distortion: float, total_max_distortion: float,
                      out_of_max_bound_reward: float,
                      out_of_min_bound_reward: float,
                      in_bounds_reward: float,
                      punish_factor: float,
                      column_types: dict,
                      action_space: ActionSpace,
                      save_distoreted_sets_dir: str,
                      use_identifying_column_dist_in_total_dist: bool,
                      use_identifying_column_dist_factor: float,
                      gamma: float,
                      n_rounds_below_min_distortion: int) -> DiscreteStateEnvironment:

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
                                                        out_of_max_bound_reward=out_of_max_bound_reward,
                                                        out_of_min_bound_reward=out_of_min_bound_reward,
                                                        in_bounds_reward=in_bounds_reward,
                                                        min_distortions=min_distortion, max_distortions=max_distortion,
                                                        punish_factor=punish_factor),
                                                    gamma=gamma,
                                                    min_distortion=min_distortion,
                                                    min_total_distortion=total_min_distortion,
                                                    max_distortion=max_distortion,
                                                    max_total_distortion=total_max_distortion,
                                                    n_rounds_below_min_distortion=n_rounds_below_min_distortion,
                                                    distorted_set_path=Path(save_distoreted_sets_dir),
                                                    n_states=n_states, env_type=env_type, column_types=column_types,
                                                    use_identifying_column_dist_in_total_dist=use_identifying_column_dist_in_total_dist,
                                                    use_identifying_column_dist_factor=use_identifying_column_dist_factor)

        return env
