import random
from pathlib import Path
import numpy as np

from src.algorithms.q_learning import QLearning, QLearnConfig
from src.spaces.env_type import DiscreteEnvType

from src.trainers.trainer import Trainer
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecayOption
from src.utils.plot_utils import plot_running_avg
from src.utils import INFO
from src.examples.helpers.load_three_columns_dataset import load_discrete_env

N_BINS = 10
N_EPISODES = 1000
OUTPUT_MSG_FREQUENCY = 100
GAMMA = 0.99
ALPHA = 0.1
N_ITRS_PER_EPISODE = 30
EPS = 1.0
EPSILON_DECAY_OPTION = EpsilonDecayOption.CONSTANT_RATE #.INVERSE_STEP
EPSILON_DECAY_FACTOR = 0.01
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.3
#OUT_OF_MAX_BOUND_REWARD = -1.0
#OUT_OF_MIN_BOUND_REWARD = -1.0
#IN_BOUNDS_REWARD = 5.0
N_ROUNDS_BELOW_MIN_DISTORTION = 10
SAVE_DISTORTED_SETS_DIR = "/home/alex/qi3/drl_anonymity/src/examples/q_learning_multistate_results/distorted_set"
#REWARD_FACTOR = 0.95
PUNISH_FACTOR = 2.0


if __name__ == '__main__':

    # set the seed for random engine
    random.seed(42)

    # load the discrete environment
    env = load_discrete_env(env_type=DiscreteEnvType.MULTI_COLUMN_STATE, n_states=N_BINS,
                            min_distortion={"ethnicity": 0.15, "salary": 0.15,
                                            "diagnosis": 0.0},
                            max_distortion={"ethnicity": 0.35, "salary": 0.35,
                                            "diagnosis": 0.0},
                            total_min_distortion=MIN_DISTORTION, total_max_distortion=MAX_DISTORTION,
                            punish_factor=PUNISH_FACTOR)
    env.config.state_type = DiscreteEnvType.MULTI_COLUMN_STATE

    # establish the configuration for the Tiled environment
    #tiled_env_config = TiledEnvConfig(n_layers=N_LAYERS, n_bins=N_BINS,
    #                                  env=discrete_env,
    #                                  column_ranges={"ethnicity": [0.0, 1.0],
    #                                                "salary": [0.0, 1.0],
    #                                                "diagnosis": [0.0, 1.0]})
    # create the Tiled environment
    #tiled_env = TiledEnv(tiled_env_config)
    #tiled_env.create_tiles()

    algo_config = QLearnConfig()
    algo_config.n_itrs_per_episode = N_ITRS_PER_EPISODE
    algo_config.gamma = GAMMA
    algo_config.alpha = ALPHA
    # algo_config.policy = SoftMaxPolicy(n_actions=len(action_space), tau=1.2)
    algo_config.policy = EpsilonGreedyPolicy(eps=EPS, n_actions=env.n_actions,
                                             decay_op=EPSILON_DECAY_OPTION,
                                             epsilon_decay_factor=EPSILON_DECAY_FACTOR)

    # the learner we want to train
    agent = QLearning(algo_config=algo_config)

    # create a trainer to train the Qlearning agent
    configuration = {"n_episodes": N_EPISODES, "output_msg_frequency": OUTPUT_MSG_FREQUENCY}
    trainer = Trainer(env=env, agent=agent, configuration=configuration)

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

    """
    # Let's play
    env.reset()

    stop_criterion = IterationControl(n_itrs=10, min_dist=MIN_DISTORTION, max_dist=MAX_DISTORTION)
    agent.play(env=env, stop_criterion=stop_criterion)
    env.save_current_dataset(episode_index=-2, save_index=False)
    """
    print("{0} Done....".format(INFO))
    print("=============================================")
