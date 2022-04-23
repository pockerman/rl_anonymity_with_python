"""
Simple example that shows how to apply QLearning
on a dataset with three columns
"""
import numpy as np
import random


from src.trainers.trainer import Trainer, TrainerConfig
from src.algorithms.q_learning import QLearning, QLearnConfig
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from src.utils.iteration_control import IterationControl
from src.examples.helpers.plot_utils import plot_running_avg
from src.datasets import ColumnType
from src.examples.helpers.load_three_columns_mock_dataset import load_discrete_env, \
    get_ethinicity_hierarchy, get_salary_bins, load_mock_subjects
from src.spaces.env_type import DiscreteEnvType
from src.utils import INFO

# configuration params
EPS = 1.0
EPSILON_DECAY_OPTION = EpsilonDecayOption.CONSTANT_RATE  # .INVERSE_STEP
EPSILON_DECAY_FACTOR = 0.01
GAMMA = 0.99
ALPHA = 0.1
N_EPISODES = 1001
N_ITRS_PER_EPISODE = 30
N_STATES = 10
# fix the rewards. Assume that any average distortion in
# (0.3, 0.7) suits us
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.3
OUT_OF_MAX_BOUND_REWARD = -1.0
OUT_OF_MIN_BOUND_REWARD = -1.0
IN_BOUNDS_REWARD = 5.0
OUTPUT_MSG_FREQUENCY = 100
N_ROUNDS_BELOW_MIN_DISTORTION = 10
SAVE_DISTORTED_SETS_DIR = "q_learning_three_columns_results/distorted_set"
PUNISH_FACTOR = 2.0


if __name__ == '__main__':

    # set the seed for random engine
    random.seed(42)

    # set the seed for random engine
    random.seed(42)

    column_types = {"ethnicity": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "salary": ColumnType.QUASI_IDENTIFYING_ATTRIBUTE,
                    "diagnosis": ColumnType.INSENSITIVE_ATTRIBUTE}

    action_space = ActionSpace(n=5)
    # all the columns that are SENSITIVE_ATTRIBUTE will be kept as they are
    # because currently we have no model
    # also INSENSITIVE_ATTRIBUTE will be kept as is
    action_space.add_many(ActionIdentity(column_name="salary"),
                          ActionIdentity(column_name="diagnosis"),
                          ActionIdentity(column_name="ethnicity"),
                          ActionStringGeneralize(column_name="ethnicity",
                                                 generalization_table=get_ethinicity_hierarchy()),
                          ActionNumericBinGeneralize(column_name="salary",
                                                     generalization_table=get_salary_bins(ds=load_mock_subjects(),
                                                                                          n_states=N_STATES)))

    env = load_discrete_env(env_type=DiscreteEnvType.TOTAL_DISTORTION_STATE, n_states=N_STATES,
                            action_space=action_space,
                            min_distortion=MIN_DISTORTION, max_distortion=MIN_DISTORTION,
                            total_min_distortion=MIN_DISTORTION, total_max_distortion=MAX_DISTORTION,
                            punish_factor=PUNISH_FACTOR, column_types=column_types,
                            save_distoreted_sets_dir=SAVE_DISTORTED_SETS_DIR,
                            use_identifying_column_dist_in_total_dist=False,
                            use_identifying_column_dist_factor=-100,
                            gamma=GAMMA,
                            in_bounds_reward=IN_BOUNDS_REWARD,
                            out_of_min_bound_reward=OUT_OF_MIN_BOUND_REWARD,
                            out_of_max_bound_reward=OUT_OF_MAX_BOUND_REWARD,
                            n_rounds_below_min_distortion=N_ROUNDS_BELOW_MIN_DISTORTION)

    # save the data before distortion so that we can
    # later load it on ARX
    env.save_current_dataset(episode_index=-1, save_index=False)

    # configuration for the Q-learner
    algo_config = QLearnConfig(gamma=GAMMA, alpha=ALPHA,
                               n_itrs_per_episode=N_ITRS_PER_EPISODE,
                               policy=EpsilonGreedyPolicy(eps=EPS, n_actions=env.n_actions,
                                                          decay_op=EPSILON_DECAY_OPTION,
                                                          epsilon_decay_factor=EPSILON_DECAY_FACTOR))

    agent = QLearning(algo_config=algo_config)

    trainer_config = TrainerConfig(n_episodes=N_EPISODES, output_msg_frequency=OUTPUT_MSG_FREQUENCY)
    trainer = Trainer(env=env, agent=agent, configuration=trainer_config)
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



