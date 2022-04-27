import random
from pathlib import Path
import numpy as np
import torch

from src.algorithms.a2c import A2C, A2CConfig
from src.networks.a2c_networks import A2CNetSimpleLinear
from src.examples.helpers.load_full_mock_dataset import load_discrete_env, get_ethinicity_hierarchy, \
    get_gender_hierarchy, get_salary_bins, load_mock_subjects
from src.datasets import ColumnType
from src.spaces.env_type import DiscreteEnvType
from src.spaces.action_space import ActionSpace
from src.spaces.actions import ActionIdentity, ActionStringGeneralize, ActionNumericBinGeneralize
from src.utils.iteration_control import IterationControl
from src.examples.helpers.plot_utils import plot_running_avg
from src.spaces.multiprocess_env import MultiprocessEnv
from src.trainers.pytorch_trainer import PyTorchTrainer, PyTorchTrainerConfig
from src.maths.optimizer_type import OptimizerType
from src.maths.pytorch_optimizer_config import PyTorchOptimizerConfig
from src.utils import INFO

N_STATES = 10
N_ITRS_PER_EPISODE = 400
ACTION_SPACE_SIZE = 10
N_WORKERS = 3
N_EPISODES = 1001
GAMMA = 0.99
ALPHA = 0.1
PUNISH_FACTOR = 2.0
MAX_DISTORTION = 0.7
MIN_DISTORTION = 0.4
SAVE_DISTORTED_SETS_DIR = "/home/alex/qi3/drl_anonymity/src/examples/a2c_all_cols_multi_state_results/distorted_set"
USE_IDENTIFYING_COLUMNS_DIST = True
IDENTIFY_COLUMN_DIST_FACTOR = 0.1
OUT_OF_MAX_BOUND_REWARD = -1.0
OUT_OF_MIN_BOUND_REWARD = -1.0
IN_BOUNDS_REWARD = 5.0
OUTPUT_MSG_FREQUENCY = 100
N_ROUNDS_BELOW_MIN_DISTORTION = 10
N_COLUMNS = 11


def env_loader(kwargs):

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
    action_space = ActionSpace(n=ACTION_SPACE_SIZE)

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
    # shuffle the action space
    # using different seeds
    action_space.shuffle(seed=kwargs["rank"] + 1)

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

    # we want to get the distances as states
    # not bin indices
    env.config.state_as_distances = True

    return env


def action_sampler(logits: torch.Tensor) -> torch.distributions.Distribution:

    action_dist = torch.distributions.Categorical(logits=logits)
    return action_dist


if __name__ == '__main__':
    # set the seed for random engine
    random.seed(42)

    # set the seed for PyTorch
    torch.manual_seed(42)

    # this the A2C network
    net = A2CNetSimpleLinear(n_columns=N_COLUMNS, n_actions=ACTION_SPACE_SIZE)

    # agent configuration
    a2c_config = A2CConfig(action_sampler=action_sampler, n_iterations_per_episode=N_ITRS_PER_EPISODE,
                           a2cnet=net, save_model_path=Path("./a2c_all_cols_multi_state_results/"),
                           n_workers=N_WORKERS,
                           normalize_advantages=True,
                           gamma=GAMMA,
                           tau=0.1,
                           beta=None, # don't use entropy
                           policy_loss_weight=1.0,
                           value_loss_weight=1.0,
                           max_grad_norm=1.0,
                           batch_size=N_ITRS_PER_EPISODE,
                           device='cpu',
                           optimizer_config=PyTorchOptimizerConfig(optimizer_type=OptimizerType.ADAM,
                                                                   optimizer_learning_rate=ALPHA))

    # create the agent
    agent = A2C(a2c_config)

    # create a trainer to train the Qlearning agent
    configuration = PyTorchTrainerConfig(n_episodes=N_EPISODES)

    # set up the arguments
    env = MultiprocessEnv(env_builder=env_loader, env_args={}, n_workers=N_WORKERS)

    try:

        env.make(agent=agent)
        trainer = PyTorchTrainer(env=env, agent=agent, config=configuration)

        # train the agent
        trainer.train()

        avg_rewards = trainer.total_rewards
        plot_running_avg(avg_rewards, steps=100,
                         xlabel="Episodes", ylabel="Reward",
                         title="Running reward average over 100 episodes")

        avg_episode_dist = np.array(trainer.total_distortions)
        print("{0} Max/Min distortion {1}/{2}".format(INFO, np.max(avg_episode_dist), np.min(avg_episode_dist)))

        plot_running_avg(avg_episode_dist, steps=100,
                         xlabel="Episodes", ylabel="Distortion",
                         title="Running distortion average over 100 episodes")

        # play the agent on the environment.
        # call the environment builder to create
        # an instance of the environment
        discrte_env = env.env_builder()

        stop_criterion = IterationControl(n_itrs=10, min_dist=MIN_DISTORTION, max_dist=MAX_DISTORTION)
        agent.play(env=discrte_env, criteria=stop_criterion)

    except Exception as e:
        print("An excpetion was thrown...{0}".format(str(e)))
    finally:
        env.close()
