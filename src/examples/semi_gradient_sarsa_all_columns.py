import random
import numpy as np

from src.algorithms.semi_gradient_sarsa import SemiGradSARSAConfig, SemiGradSARSA
from src.spaces.tiled_environment import TiledEnv, TiledEnvConfig, Layer

from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.trainers.trainer import Trainer, TrainerConfig
from src.policies.epsilon_greedy_policy import EpsilonDecayOption
from src.algorithms.epsilon_greedy_q_estimator import EpsilonGreedyQEstimatorConfig, EpsilonGreedyQEstimator
from src.datasets import ColumnType
from src.spaces.env_type import DiscreteEnvType
from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, \
    get_gender_hierarchy, get_salary_bins, load_mock_subjects
from src.examples.helpers.plot_utils import plot_running_avg
from src.utils.iteration_control import IterationControl
from src.utils import INFO

N_STATES = 10
N_LAYERS = 5
N_BINS = 10
N_EPISODES = 1001
OUTPUT_MSG_FREQUENCY = 100
GAMMA = 0.99
ALPHA = 0.1
N_ITRS_PER_EPISODE = 30
EPS = 1.0
EPSILON_DECAY_OPTION = EpsilonDecayOption.CONSTANT_RATE
EPSILON_DECAY_FACTOR = 0.01
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.4
OUT_OF_MAX_BOUND_REWARD = -1.0
OUT_OF_MIN_BOUND_REWARD = -1.0
IN_BOUNDS_REWARD = 5.0
N_ROUNDS_BELOW_MIN_DISTORTION = 10
SAVE_DISTORTED_SETS_DIR = "semi_grad_sarsa_all_columns/distorted_set"
PUNISH_FACTOR = 2.0
USE_IDENTIFYING_COLUMNS_DIST = True
IDENTIFY_COLUMN_DIST_FACTOR = 0.1


"""
def load_discrete_env() -> DiscreteStateEnvironment:

        mock_ds = load_mock_subjects()

        # create bins for the salary generalization
        unique_salary = mock_ds.get_column_unique_values(col_name="salary")
        unique_salary.sort()

        # modify slightly the max value because
        # we get out of bounds for the maximum salary
        bins = np.linspace(unique_salary[0], unique_salary[-1] + 1, N_BINS)

        action_space = ActionSpace(n=5)
        action_space.add_many(ActionIdentity(column_name="ethnicity"),
                              ActionStringGeneralize(column_name="ethnicity",
                                                     generalization_table=get_ethinicity_hierarchy()),
                              ActionIdentity(column_name="salary"),
                              ActionNumericBinGeneralize(column_name="salary", generalization_table=bins),
                              ActionIdentity(column_name="diagnosis"))

        action_space.shuffle()

        env = DiscreteStateEnvironment.from_options(data_set=mock_ds,
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

        return env
"""

if __name__ == '__main__':

    # set the seed for random engine
    random.seed(42)

    # specify the column types. An identifying column
    # will me removed from the anonymized data set
    # An  INSENSITIVE_ATTRIBUTE remains intact.
    # A QUASI_IDENTIFYING_ATTRIBUTE is used in the anonymization
    # A SENSITIVE_ATTRIBUTE currently remains intact
    column_types = {"NHSno": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "given_name": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "surname": ColumnType.IDENTIFYING_ATTRIBUTE,
                    "gender": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "dob": ColumnType.SENSITIVE_ATTRIBUTE,
                    "ethnicity": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "education": ColumnType.SENSITIVE_ATTRIBUTE,
                    "salary": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "mutation_status": ColumnType.SENSITIVE_ATTRIBUTE,
                    "preventative_treatment": ColumnType.SENSITIVE_ATTRIBUTE,
                    "diagnosis": ColumnType.INSENSITIVE_ATTRIBUTE}

    # define the action space
    action_space = ActionSpace(n=10)

    # all the columns that are SENSITIVE_ATTRIBUTE will be kept as they are
    # because currently we have no model
    # also INSENSITIVE_ATTRIBUTE will be kept as is
    # in order to declare this we use an ActionIdentity
    action_space.add_many(ActionIdentity(column_name="dob"),
                          ActionIdentity(column_name="education"),
                          ActionIdentity(column_name="salary"),
                          ActionIdentity(column_name="diagnosis"),
                          ActionIdentity(column_name="mutation_status"),
                          ActionIdentity(column_name="preventative_treatment"),
                          ActionIdentity(column_name="ethnicity"),
                          ActionStringGeneralize(column_name="ethnicity",
                                                 generalization_table=get_ethinicity_hierarchy()),
                          ActionStringGeneralize(column_name="gender",
                                                 generalization_table=get_gender_hierarchy()),
                          ActionNumericBinGeneralize(column_name="salary",
                                                     generalization_table=get_salary_bins(ds=load_mock_subjects(),
                                                                                          n_states=N_STATES)))

    action_space.shuffle()

    # load the discrete environment
    env = load_discrete_env(env_type=DiscreteEnvType.MULTI_COLUMN_STATE, n_states=N_STATES,
                            min_distortion={"ethnicity": 0.133, "salary": 0.133, "gender": 0.133,
                                            "dob": 0.0, "education": 0.0, "diagnosis": 0.0,
                                            "mutation_status": 0.0, "preventative_treatment": 0.0,
                                            "NHSno": 0.0, "given_name": 0.0, "surname": 0.0},
                            max_distortion={"ethnicity": 0.133, "salary": 0.133, "gender": 0.133,
                                            "dob": 0.0, "education": 0.0, "diagnosis": 0.0,
                                            "mutation_status": 0.0, "preventative_treatment": 0.0,
                                            "NHSno": 0.1, "given_name": 0.1, "surname": 0.1},
                            total_min_distortion=MIN_DISTORTION, total_max_distortion=MAX_DISTORTION,
                            out_of_max_bound_reward=OUT_OF_MAX_BOUND_REWARD,
                            out_of_min_bound_reward=OUT_OF_MIN_BOUND_REWARD,
                            in_bounds_reward=IN_BOUNDS_REWARD,
                            punish_factor=PUNISH_FACTOR,
                            column_types=column_types,
                            action_space=action_space,
                            save_distoreted_sets_dir=SAVE_DISTORTED_SETS_DIR,
                            use_identifying_column_dist_in_total_dist=USE_IDENTIFYING_COLUMNS_DIST,
                            use_identifying_column_dist_factor=IDENTIFY_COLUMN_DIST_FACTOR,
                            gamma=GAMMA,
                            n_rounds_below_min_distortion=N_ROUNDS_BELOW_MIN_DISTORTION)

    # establish the configuration for the Tiled environment
    tiled_env_config = TiledEnvConfig(n_layers=N_LAYERS, n_bins=N_BINS,
                                      env=env,
                                      column_ranges={"gender": [0.0, 1.0],
                                                     "ethnicity": [0.0, 1.0],
                                                     "salary": [0.0, 1.0]})

    # create the Tiled environment
    tiled_env = TiledEnv(tiled_env_config)
    tiled_env.create_tiles()

    # agent configuration
    agent_config = SemiGradSARSAConfig(gamma=GAMMA, alpha=ALPHA, n_itrs_per_episode=N_ITRS_PER_EPISODE,
                                       policy=EpsilonGreedyQEstimator(EpsilonGreedyQEstimatorConfig(eps=EPS,
                                                                                                    n_actions=tiled_env.n_actions,
                                                                                                    decay_op=EPSILON_DECAY_OPTION,
                                                                                                    epsilon_decay_factor=EPSILON_DECAY_FACTOR,
                                                                                                    env=tiled_env,
                                                                                                    gamma=GAMMA,
                                                                                                    alpha=ALPHA)))
    # create the agent
    agent = SemiGradSARSA(agent_config)

    # create a trainer to train the SemiGradSARSA agent
    trainer_config = TrainerConfig(n_episodes=N_EPISODES, output_msg_frequency=OUTPUT_MSG_FREQUENCY)
    trainer = Trainer(env=tiled_env, agent=agent, configuration=trainer_config)

    # train the agent
    trainer.train()

    # avg_rewards = trainer.avg_rewards()
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
    # Let's play
    tiled_env.reset()

    stop_criterion = IterationControl(n_itrs=10, min_dist=MIN_DISTORTION, max_dist=MAX_DISTORTION)
    agent.play(env=tiled_env, stop_criterion=stop_criterion)
    tiled_env.save_current_dataset(episode_index=-2, save_index=False)
    print("{0} Done....".format(INFO))
    print("=============================================")

