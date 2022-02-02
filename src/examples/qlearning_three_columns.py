"""
Simple example that shows how to apply QLearning
on a dataset with three columns
"""
import numpy as np
import random
from pathlib import Path

from src.algorithms.q_learning import QLearning, QLearnConfig
from src.algorithms.trainer import Trainer
from src.datasets.datasets_loaders import MockSubjectsLoader
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.utils.reward_manager import RewardManager
from src.utils.serial_hierarchy import SerialHierarchy
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption
from src.policies.softmax_policy import SoftMaxPolicy
from src.utils.numeric_distance_type import NumericDistanceType
from src.utils.string_distance_calculator import StringDistanceType
from src.utils.distortion_calculator import DistortionCalculationType, DistortionCalculator
from src.spaces.discrete_state_environment import DiscreteStateEnvironment, DiscreteEnvConfig
from src.utils.iteration_control import IterationControl
from src.utils.plot_utils import plot_running_avg
from src.utils import INFO


def get_ethinicity_hierarchy():
    ethnicity_hierarchy = SerialHierarchy(values={})
    ethnicity_hierarchy.add("Mixed White/Asian", "White/Asian")
    ethnicity_hierarchy.add("White/Asian", "White")

    ethnicity_hierarchy.add("Chinese", "Asian")
    ethnicity_hierarchy.add("Indian", "Asian")

    ethnicity_hierarchy.add("Mixed White/Black African", "African-Mixed")
    ethnicity_hierarchy.add("African-Mixed", "Mixed")

    ethnicity_hierarchy.add("Black African", "African")
    ethnicity_hierarchy.add("African", "African")

    ethnicity_hierarchy.add("Asian other", "Asian")
    ethnicity_hierarchy.add("Black other", "Black")

    ethnicity_hierarchy.add("Mixed White/Black Caribbean", "Caribbean-Mixed")
    ethnicity_hierarchy.add("Caribbean-Mixed", "Mixed")

    ethnicity_hierarchy.add("Mixed other", "Mixed")
    ethnicity_hierarchy.add("Arab", "Asian")

    ethnicity_hierarchy.add("White Irish", "European-White")
    ethnicity_hierarchy.add("European-White", "European")

    ethnicity_hierarchy.add("Not stated", "Not stated")
    ethnicity_hierarchy.add("White Gypsy/Traveller", "White")

    ethnicity_hierarchy.add("White British", "British")
    ethnicity_hierarchy.add("British", "European")

    ethnicity_hierarchy.add("Bangladeshi", "Asian")
    ethnicity_hierarchy.add("White other", "White")
    ethnicity_hierarchy.add("Black Caribbean", "Black")
    ethnicity_hierarchy.add("Pakistani", "Asian")

    ethnicity_hierarchy.add("White", "White")
    ethnicity_hierarchy.add("Mixed", "Mixed")
    ethnicity_hierarchy.add("European", "European")
    ethnicity_hierarchy.add("Asian", "Asian")
    ethnicity_hierarchy.add("Black", "Black")
    ethnicity_hierarchy.add("Not stated", "Not stated")

    return ethnicity_hierarchy


if __name__ == '__main__':
    random.seed(42)

    # configuration params
    EPS = 1.0
    EPSILON_DECAY_OPTION = EpsilonDecreaseOption.CONSTANT_RATE #.INVERSE_STEP
    EPSILON_DECAY_FACTOR = 0.01
    GAMMA = 0.99
    ALPHA = 0.1
    N_EPISODES = 1001
    N_ITRS_PER_EPISODE = 30
    N_STATES = 10
    # fix the rewards. Assume that any average distortion in
    # (0.4, 0.7) suits us
    MAX_DISTORTION = 0.7
    MIN_DISTORTION = 0.4
    OUT_OF_MAX_BOUND_REWARD = -1.0
    OUT_OF_MIN_BOUND_REWARD = -1.0
    IN_BOUNDS_REWARD = 5.0
    OUTPUT_MSG_FREQUENCY = 100
    N_ROUNDS_BELOW_MIN_DISTORTION = 10
    SAVE_DISTORTED_SETS_DIR = "/home/alex/qi3/drl_anonymity/src/examples/q_learn_distorted_sets/distorted_set"

    # specify the columns to drop
    drop_columns = MockSubjectsLoader.FEATURES_DROP_NAMES + ["preventative_treatment", "gender",
                                                             "education", "mutation_status"]
    MockSubjectsLoader.FEATURES_DROP_NAMES = drop_columns

    # do a salary normalization
    MockSubjectsLoader.NORMALIZED_COLUMNS = ["salary"]

    # specify the columns to use
    MockSubjectsLoader.COLUMNS_TYPES = {"ethnicity": str, "salary": float, "diagnosis": int}
    ds = MockSubjectsLoader()

    assert ds.n_columns == 3, "Invalid number of columns {0} not equal to 3".format(ds.n_columns)

    # create bins for the salary generalization
    unique_salary = ds.get_column_unique_values(col_name="salary")
    unique_salary.sort()

    # modify slightly the max value because
    # we get out of bounds
    bins = np.linspace(unique_salary[0], unique_salary[-1] + 1, N_STATES)

    # establish the action space. For every column
    # we assume three actions except for the ```diagnosis```
    # which we do not alter
    action_space = ActionSpace(n=5)
    action_space.add_many(ActionIdentity(column_name="ethnicity"),
                          ActionStringGeneralize(column_name="ethnicity",
                                                 generalization_table=get_ethinicity_hierarchy()),
                          ActionIdentity(column_name="salary"),
                          ActionNumericBinGeneralize(column_name="salary", generalization_table=bins),
                          ActionIdentity(column_name="diagnosis"))

    action_space.shuffle()

    env_config = DiscreteEnvConfig()

    env_config.action_space = action_space
    env_config.reward_manager = RewardManager(bounds=(MIN_DISTORTION, MAX_DISTORTION),
                                              out_of_max_bound_reward=OUT_OF_MAX_BOUND_REWARD,
                                              out_of_min_bound_reward=OUT_OF_MIN_BOUND_REWARD,
                                              in_bounds_reward=IN_BOUNDS_REWARD)
    env_config.data_set = ds
    env_config.gamma = GAMMA
    env_config.max_distortion = MAX_DISTORTION
    env_config.min_distortion = MIN_DISTORTION
    env_config.n_states = N_STATES
    env_config.n_rounds_below_min_distortion = N_ROUNDS_BELOW_MIN_DISTORTION
    env_config.distorted_set_path = Path(SAVE_DISTORTED_SETS_DIR)
    env_config.distortion_calculator = DistortionCalculator(
        numeric_column_distortion_metric_type=NumericDistanceType.L2_AVG,
        string_column_distortion_metric_type=StringDistanceType.COSINE_NORMALIZE,
        dataset_distortion_type=DistortionCalculationType.SUM)

    # create the environment
    env = DiscreteStateEnvironment(env_config=env_config)
    env.reset()

    # save the data before distortion so that we can
    # later load it on ARX
    env.save_current_dataset(episode_index=-1, save_index=False)

    # configuration for the Q-learner
    algo_config = QLearnConfig()
    algo_config.n_itrs_per_episode = N_ITRS_PER_EPISODE
    algo_config.gamma = GAMMA
    algo_config.alpha = ALPHA
    #algo_config.policy = SoftMaxPolicy(n_actions=len(action_space), tau=1.2)
    algo_config.policy = EpsilonGreedyPolicy(eps=EPS, env=env,decay_op=EPSILON_DECAY_OPTION,
                                             epsilon_decay_factor=EPSILON_DECAY_FACTOR)

    # the learner we want to train
    agent = QLearning(algo_config=algo_config)

    configuration = {"n_episodes": N_EPISODES, "output_msg_frequency": OUTPUT_MSG_FREQUENCY}

    # create a trainer to train the Qlearning agent
    trainer = Trainer(env=env, agent=agent, configuration=configuration)
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
    env.reset()

    stop_criterion = IterationControl(n_itrs=10, min_dist=MIN_DISTORTION, max_dist=MAX_DISTORTION)
    agent.play(env=env, stop_criterion=stop_criterion)
    env.save_current_dataset(episode_index=-2, save_index=False)
    print("{0} Done....".format(INFO))
    print("=============================================")

