import matplotlib.pyplot as plt
import numpy as np

from src.algorithms.q_learning import QLearning, QLearnConfig
from src.algorithms.trainer import Trainer
from src.utils.string_distance_calculator import StringDistanceType
from src.spaces.actions import ActionSuppress, ActionIdentity, ActionGeneralize, ActionTransform
from src.spaces.environment import Environment, EnvConfig
from src.spaces.action_space import ActionSpace
from src.datasets.datasets_loaders import MockSubjectsLoader
from src.utils.reward_manager import RewardManager
from src.policies.epsilon_greedy_policy import EpsilonGreedyPolicy, EpsilonDecreaseOption
from src.utils.serial_hierarchy import SerialHierarchy
from src.utils.numeric_distance_type import NumericDistanceType


def plot_running_avg(avg_rewards):

    running_avg = np.empty(avg_rewards.shape[0])
    for t in range(avg_rewards.shape[0]):
        running_avg[t] = np.mean(avg_rewards[max(0, t-100) : (t+1)])
    plt.plot(running_avg)
    plt.xlabel("Number of episodes")
    plt.ylabel("Reward")
    plt.title("Running average")
    plt.show()

def get_ethinicity_hierarchies():

    ethnicity_hierarchy = SerialHierarchy()
    ethnicity_hierarchy.add("Mixed White/Asian", values=["Mixed", '*'])
    ethnicity_hierarchy.add("Chinese", values=["Asian", '*'])
    ethnicity_hierarchy.add("Indian", values=["Asian", '*'])
    ethnicity_hierarchy.add("Mixed White/Black African", values=["Mixed", '*'])
    ethnicity_hierarchy.add("Black African", values=["Black", '*'])
    ethnicity_hierarchy.add("Asian other", values=["Asian", "*"])
    ethnicity_hierarchy.add("Black other", values=["Black", "*"])
    ethnicity_hierarchy.add("Mixed White/Black Caribbean", values=["Mixed", "*"])
    ethnicity_hierarchy.add("Mixed other", values=["Mixed", "*"])
    ethnicity_hierarchy.add("Arab", values=["Asian", "*"])
    ethnicity_hierarchy.add("White Irish", values=["White", "*"])
    ethnicity_hierarchy.add("Not stated", values=["Not stated", "*"])
    ethnicity_hierarchy.add("White Gypsy/Traveller", values=["White", "*"])
    ethnicity_hierarchy.add("White British", values=["White", "*"])
    ethnicity_hierarchy.add("Bangladeshi", values=["Asian", "*"])
    ethnicity_hierarchy.add("White other", values=["White", "*"])
    ethnicity_hierarchy.add("Black Caribbean", values=["Black", "*"])
    ethnicity_hierarchy.add("Pakistani", values=["Asian", "*"])

    return ethnicity_hierarchy


if __name__ == '__main__':

    EPS = 1.0
    GAMMA = 0.99
    ALPHA = 0.1
    N_EPISODES = 100

    # load the dataset
    ds = MockSubjectsLoader()

    # generalization table for the ethnicity column
    ethinicity_table = get_ethinicity_hierarchies()

    # specify the action space. We need to establish how these actions
    # are performed
    action_space = ActionSpace(n=5)
    action_space.add_many(ActionSuppress(column_name="gender", suppress_table={"F": SerialHierarchy(values=['*', ]),
                                                                               'M': SerialHierarchy(values=['*', ])}),
                          ActionIdentity(column_name="salary"),
                          ActionIdentity(column_name="education"),
                          ActionGeneralize(column_name="ethnicity", generalization_table=ethinicity_table),
                          ActionSuppress(column_name="preventative_treatment",
                                         suppress_table={"No":  SerialHierarchy(values=['Maybe', '*']),
                                                         'Yes': SerialHierarchy(values=['Maybe', '*']),
                                                         "NA":  SerialHierarchy(values=['Maybe', '*']),
                                                         "Maybe": SerialHierarchy(values=['*', '*'])
                                                         }))

    # average distirtion
    average_distortion_constraint = {"salary": [0.0, 0.0, 0.0], "education": [0.0, 0.0, 0.0],
                                     "ethnicity": [3.0, 1.0, -1.0], "gender": [4.0, 1.0, -1.0],
                                     "preventative_treatment": [4.0, 1.0, -1.0]}

    # specify the reward manager to use
    reward_manager = RewardManager(average_distortion_constraint=average_distortion_constraint)

    env_config = EnvConfig()
    env_config.start_column = "gender"
    env_config.action_space = action_space
    env_config.reward_manager = reward_manager
    env_config.data_set = ds
    env_config.gamma = 0.99
    env_config.numeric_column_distortion_metric_type = NumericDistanceType.L2

    # create the environment
    env = Environment(env_config=env_config)

    # initialize text distances
    env.initialize_text_distances(distance_type=StringDistanceType.COSINE)

    algo_config = QLearnConfig()
    algo_config.n_itrs_per_episode = 10
    algo_config.gamma = 0.99
    algo_config.alpha = 0.1
    algo_config.policy = EpsilonGreedyPolicy(eps=EPS, env=env,
                                             decay_op=EpsilonDecreaseOption.INVERSE_STEP)

    agent = QLearning(algo_config=algo_config)

    configuration = {"n_episodes": N_EPISODES, "output_msg_frequency": 10}

    # create a trainer to train the A2C agent
    trainer = Trainer(env=env, agent=agent, configuration=configuration)

    trainer.train()

    # get the state space
    state_space = env.state_space

    for state in state_space:
        print("Column {0} history {1}".format(state, state_space[state].history))

    total_reward = trainer.total_rewards
    episodes = [episode for episode in range(N_EPISODES)]

    plt.plot(episodes, total_reward)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()


