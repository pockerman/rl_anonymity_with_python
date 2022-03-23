import random
from pathlib import Path
import numpy as np
import torch

from src.algorithms.a2c import A2C, A2CConfig
from src.networks.a2c_networks import A2CNetSimpleLinear
from src.utils.serial_hierarchy import SerialHierarchy
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig, Layer
from src.spaces.discrete_state_environment import DiscreteStateEnvironment
from src.datasets.datasets_loaders import MockSubjectsLoader, MockSubjectsData
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.policies.epsilon_greedy_policy import EpsilonDecayOption
from src.utils.distortion_calculator import DistortionCalculationType, DistortionCalculator
from src.maths.numeric_distance_type import NumericDistanceType
from src.maths.string_distance_calculator import StringDistanceType
from src.utils.reward_manager import RewardManager
from src.algorithms.pytorch_multi_process_trainer import PyTorchMultiProcessTrainer, PyTorchMultiProcessTrainerConfig, OptimizerConfig
from src.utils import INFO


N_LAYERS = 1
N_BINS = 10
N_EPISODES = 10
OUTPUT_MSG_FREQUENCY = 100
GAMMA = 0.99
ALPHA = 0.1
N_ITRS_PER_EPISODE = 30
EPS = 1.0
EPSILON_DECAY_OPTION = EpsilonDecayOption.CONSTANT_RATE #.INVERSE_STEP
EPSILON_DECAY_FACTOR = 0.01
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.3
OUT_OF_MAX_BOUND_REWARD = -1.0
OUT_OF_MIN_BOUND_REWARD = -1.0
IN_BOUNDS_REWARD = 5.0
N_ROUNDS_BELOW_MIN_DISTORTION = 10
SAVE_DISTORTED_SETS_DIR = "/home/alex/qi3/drl_anonymity/src/examples/semi_grad_sarsa/distorted_set"
REWARD_FACTOR = 0.95
PUNISH_FACTOR = 2.0
ACTION_SPACE_SIZE = 5


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

    mock_data = MockSubjectsData(FILENAME=Path("../../data/mocksubjects.csv"),
                                 COLUMNS_TYPES={"ethnicity": str, "salary": float, "diagnosis": int},
                                 FEATURES_DROP_NAMES=["NHSno", "given_name",
                                                      "surname", "dob"] + ["preventative_treatment",
                                                                           "gender", "education", "mutation_status"],
                                 NORMALIZED_COLUMNS=["salary"])

    ds = MockSubjectsLoader(mock_data)

    assert ds.n_columns == 3, "Invalid number of columns {0} not equal to 3".format(ds.n_columns)

    return ds


def load_discrete_env() -> DiscreteStateEnvironment:

        mock_ds = load_mock_subjects()

        # create bins for the salary generalization
        unique_salary = mock_ds.get_column_unique_values(col_name="salary")
        unique_salary.sort()

        # modify slightly the max value because
        # we get out of bounds for the maximum salary
        bins = np.linspace(unique_salary[0], unique_salary[-1] + 1, N_BINS)

        action_space = ActionSpace(n=ACTION_SPACE_SIZE)
        action_space.add_many(ActionIdentity(column_name="ethnicity"),
                              ActionStringGeneralize(column_name="ethnicity",
                                                     generalization_table=get_ethinicity_hierarchy()),
                              ActionIdentity(column_name="salary"),
                              ActionNumericBinGeneralize(column_name="salary", generalization_table=bins),
                              ActionIdentity(column_name="diagnosis"))

        action_space.shuffle()

        discrete_env = DiscreteStateEnvironment.from_options(data_set=mock_ds,
                                                    action_space=action_space,
                                                    distortion_calculator=DistortionCalculator(
                                                        numeric_column_distortion_metric_type=NumericDistanceType.L2_AVG,
                                                        string_column_distortion_metric_type=StringDistanceType.COSINE_NORMALIZE,
                                                        dataset_distortion_type=DistortionCalculationType.SUM),
                                                    reward_manager=RewardManager(bounds=(MIN_DISTORTION, MAX_DISTORTION),
                                                                                 out_of_max_bound_reward=OUT_OF_MAX_BOUND_REWARD,
                                                                                 out_of_min_bound_reward=OUT_OF_MIN_BOUND_REWARD,
                                                                                 in_bounds_reward=IN_BOUNDS_REWARD),
                                                    gamma=GAMMA,
                                                    reward_factor=REWARD_FACTOR,
                                                    punish_factor=PUNISH_FACTOR,
                                                    min_distortion=MIN_DISTORTION, max_distortion=MAX_DISTORTION,
                                                    n_rounds_below_min_distortion=N_ROUNDS_BELOW_MIN_DISTORTION,
                                                    distorted_set_path=Path(SAVE_DISTORTED_SETS_DIR),
                                                    n_states=N_LAYERS * Layer.n_tiles_per_action(N_BINS,
                                                                                                 mock_ds.n_columns))

        # establish the configuration for the Tiled environment
        tiled_env_config = TiledEnvConfig(n_layers=N_LAYERS, n_bins=N_BINS,
                                          env=discrete_env,
                                          column_ranges={"ethnicity": [0.0, 1.0],
                                                         "salary": [0.0, 1.0],
                                                         "diagnosis": [0.0, 1.0]})
        # create the Tiled environment
        tiled_env = TiledEnv(tiled_env_config)
        tiled_env.create_tiles()

        return tiled_env


def action_sampler(logits) -> torch.Tensor:

    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    return action


if __name__ == '__main__':

    # set the seed for random engine
    random.seed(42)

    net = A2CNetSimpleLinear(n_columns=3, n_actions=ACTION_SPACE_SIZE)

    # agent configuration
    a2c_config = A2CConfig(action_sampler=action_sampler, n_iterations_per_episode=N_ITRS_PER_EPISODE,
                           a2cnet=net, save_model_path=Path("./a2c_three_columns_output/"))

    # create the agent
    agent = A2C(a2c_config)

    # create a trainer to train the Qlearning agent
    configuration = PyTorchMultiProcessTrainerConfig(n_episodes=N_EPISODES,
                                                     env_loader=load_discrete_env,
                                                     optimizer_config=OptimizerConfig())

    trainer = PyTorchMultiProcessTrainer(agent=agent, config=configuration)

    # train the agent
    trainer.train()

    # avg_rewards = trainer.avg_rewards()
    """
    avg_rewards = trainer.total_rewards
    plot_running_avg(avg_rewards, steps=100,
                     xlabel="Episodes", ylabel="Reward",
                     title="Running reward average over 100 episodes")

    avg_episode_dist = np.array(trainer.total_distortions)
    print("{0} Max/Min distortion {1}/{2}".format(INFO, np.max(avg_episode_dist), np.min(avg_episode_dist)))

    plot_running_avg(avg_episode_dist, steps=100,
                     xlabel="Episodes", ylabel="Distortion",
                     title="Running distortion average over 100 episodes")

    print("=============================================")
    print("{0} Generating distorted dataset".format(INFO))
    """

    """
    # Let's play
    env.reset()

    stop_criterion = IterationControl(n_itrs=10, min_dist=MIN_DISTORTION, max_dist=MAX_DISTORTION)
    agent.play(env=env, stop_criterion=stop_criterion)
    env.save_current_dataset(episode_index=-2, save_index=False)
    """
    print("{0} Done....".format(INFO))
    print("=============================================")
