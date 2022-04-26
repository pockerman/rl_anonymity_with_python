"""Module pytorch_multi_process_trainer. Specifies a trainer
for PyTorch-based models.

"""

import numpy as np
import torch.nn as nn
from typing import TypeVar, Any
from dataclasses import dataclass



from src.utils import INFO
from src.utils.function_wraps import time_func, time_func_wrapper
from src.utils.episode_info import EpisodeInfo
from src.maths.optimizer_type import OptimizerType
from src.maths.pytorch_optimizer_builder import pytorch_optimizer_builder
from src.utils import INFO

Env = TypeVar("Env")
Agent = TypeVar("Agent")
EnvLoader = TypeVar('EnvLoader')


@dataclass(init=True, repr=True)
class PyTorchTrainerConfig(object):
    """Configuration for PyTorchMultiProcessTrainer

    """
    n_procs: int = 1
    n_episodes: int = 100


@dataclass(init=True, repr=True)
class WorkerResult(object):
    worker_idx: int


def worker(worker_idx: int, worker_model: nn.Module, params: dir):
    """Executes the process work

    Parameters
    ----------

    worker_idx: The id of the worker
    worker_model: The model the worker is using
    params: Parameters needed

    Returns
    -------

    """

    # load the environment. Every worker has a distinct
    # copy of the environment
    env = params["env_loader"]()

    # create the optimizer
    optimizer = pytorch_optimizer_builder(opt_type=params["optimizer_config"]["optimizer_type"],
                                          model_params=worker_model.parameters(),
                                          **params["optimizer_config"])

    worker_model.set_train_mode()
    for episode in range(params["n_episodes"]):

        if worker_idx == params["master_process"]:
            print("{0} On episode {1}/{2}".format(INFO, episode, params["n_episodes"]))

        optimizer.zero_grad()

        # run the episode
        episode_info = worker_model.on_episode(env=env, episode_idx=episode, buffer_size=params["buffer_size"])

        if worker_idx == params["master_process"]:
            print("{0} Episode {1} finished in {2} secs".format(INFO, episode, episode_info.total_execution_time))
            print("{0} Episode score={1}, episode total avg distortion {2}".format(INFO, episode_info.episode_score,
                                                                                   episode_info.total_distortion / episode_info.episode_itrs))

            print("{0} Episode finished after {1} iterations".format(INFO, episode_info.episode_itrs))

        # update the parameters this function is moel dependent
        # so should be more generic
        params["update_params_functor"](optimizer, episode_info, config=worker_model.config) #**params)


class PyTorchTrainer(object):
    """The class PyTorchMultiProcessTrainer. Trainer
    for multiprocessing with PyTorch

    """

    def __init__(self, env: Env, agent: Agent, config: PyTorchTrainerConfig) -> None:
        """Constructor. Initialize a trainer by passing the training environment
        instance the agent to train and configuration dictionary

        Parameters
        ----------

        agent: The agent to train
        config: Configuration parameters for the trainer

        """

        self.env: Env = env
        self.agent = agent
        self.configuration = config
        # monitor performance
        self.total_rewards: np.array = np.zeros(self.configuration.n_episodes)
        self.iterations_per_episode = []
        self.total_distortions = np.zeros(self.configuration.n_episodes)

    def avg_rewards(self) -> np.array:
        """
        Returns the average reward per episode
        :return:
        """
        avg = np.zeros(self.configuration['n_episodes'])

        for i in range(self.total_rewards.shape[0]):
            avg[i] = self.total_rewards[i] / self.iterations_per_episode[i]
        return avg

    def avg_distortion(self) -> np.array:
        """
        Returns the average reward per episode
        :return:
        """
        avg = np.zeros(self.configuration['n_episodes'])

        for i in range(len(self.total_distortions)):
            avg[i] = self.total_distortions[i] / self.iterations_per_episode[i]
        return avg

    def actions_before_training(self) -> None:
        """Any actions to perform before training begins

        Returns
        -------

        None
        """

        if self.env is None:
            raise ValueError("Environment has not been specified")

        if self.agent is None:
            raise ValueError("Agent has not been specified")

        self.total_rewards: np.array = np.zeros(self.configuration.n_episodes)
        self.iterations_per_episode = []
        self.total_distortions = np.zeros(self.configuration.n_episodes)

        self.env.reset()
        self.agent.actions_before_training_begins(self.env)

    def actions_before_episode_begins(self, env: Env, episode_idx: int,  **options) -> None:
        """Perform any actions necessary before the training begins

        Parameters
        ----------
        env: The environment to train on
        episode_idx: The training episode index
        options: Any options passed by the client code

        Returns
        -------

        None

        """
        self.agent.actions_before_episode_begins(env, episode_idx, **options)

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:
        """Any actions after the training episode ends

        Parameters
        ----------

        env: The environment to train on
        episode_idx:  The training episode index
        options: Any options passed by the client code

        Returns
        -------

        None
        """
        self.agent.actions_after_episode_ends(env, episode_idx, **options)

        if episode_idx % self.configuration['output_msg_frequency'] == 0:
            if self.env.config.distorted_set_path is not None:
                self.env.save_current_dataset(episode_idx)

    @time_func_wrapper(show_time=True)
    def train(self):

        print("{0} Training agent {1}".format(INFO, self.agent.name))
        print("{0} Number of training episodes {1}".format(INFO, self.configuration.n_episodes))
        print("{0} Number of processes {1}".format(INFO, self.configuration.n_procs))

        # any actions needed before training starts
        self.actions_before_training()

        # create the processes by attaching the worker

        for episode in range(0, self.configuration.n_episodes):
            print("{0} On episode {1}/{2}".format(INFO, episode, self.configuration.n_episodes))

            self.actions_before_episode_begins(self.env, episode)

            # train for a number of iterations
            episode_info: EpisodeInfo = self.agent.on_episode(self.env, episode)

            print("{0} Episode {1} finished in {2} secs".format(INFO, episode, episode_info.total_execution_time))
            print("{0} Episode score={1}, episode total "
                  "avg distortion {2}".format(INFO, episode_info.episode_score,
                                              episode_info.total_distortion / episode_info.episode_itrs))

            self.iterations_per_episode.append(episode_info.episode_itrs)
            self.total_rewards[episode] = episode_info.episode_score
            self.total_distortions[episode] = episode_info.total_distortion / episode_info.episode_itrs
            self.agent.actions_after_episode_ends(self.env, episode, **{"episode_info": episode_info})

        self.actions_after_training()

    def actions_after_training(self) -> None:
        self.agent.actions_after_training()



