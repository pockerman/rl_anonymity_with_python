"""Module pytorch_multi_process_trainer. Specifies a trainer
for PyTorch-based models.

"""

import numpy as np
import torch.nn as nn
from typing import TypeVar, Any
from dataclasses import dataclass

import torch.multiprocessing as mp

from src.utils import INFO
from src.utils.function_wraps import time_func, time_func_wrapper
from src.utils.episode_info import EpisodeInfo
from src.maths.optimizer_type import OptimizerType
from src.maths.pytorch_optimizer_builder import pytorch_optimizer_builder

Env = TypeVar("Env")
Agent = TypeVar("Agent")


@dataclass(init=True, repr=True)
class PyTorchMultiProcessTrainerConfig(object):
    """Configuration for PyTorchMultiProcessTrainer

    """
    n_procs = 1
    n_episodes: 100
    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 0.01


class TorchProcsHandler(object):
    """The TorchProcsHandler class. Utility
    class to handle PyTorch processe

    """

    def __init__(self, n_procs: int) -> None:
        """Constructor

        Parameters
        ----------
        n_procs: The number of processes to handle

        """
        self.n_procs = n_procs
        self.processes = []

    def create_and_start(self, target: Any, *args) -> None:
        for i in range(self.n_procs):
            p = mp.Process(target=target, args=args)
            p.start()
            self.processes.append(p)

    def create_process_and_start(self, target: Any, args) -> None:
        p = mp.Process(target=target, args=args)
        p.start()
        self.processes.append(p)

    def join(self) -> None:
        for p in self.processes:
            p.join()

    def terminate(self) -> None:
        for p in self.processes:
            p.terminate()

    def join_and_terminate(self):
        self.join()
        self.terminate()


def worker(worker_idx: int, worker_model: nn.Module, params: dir):

        # load the environment. Every worker has a distinct
        # copy of the environment
        env = None

        # create the optimizer
        optimizer = pytorch_optimizer_builder(opt_type=params["optimizer_type"],
                                              model_params=worker_model.parameters(),
                                              **{"learning_rate": params["learning_rate"]})

        # run the training for the worker
        pass


class PyTorchMultiProcessTrainer(object):
    """The class PyTorchMultiProcessTrainer. Trainer
    for multiprocessing with PyTorch

    """

    def __init__(self, agent: Agent, config: PyTorchMultiProcessTrainerConfig) -> None:
        """Constructor. Initialize a trainer by passing the training environment
        instance the agen to train and configuration dictionary

        Parameters
        ----------

        agent: The agent to train
        config: Configuration parameters for the trainer

        """

        self.agent = agent
        self.configuration = config
        # monitor performance
        self.total_rewards: np.array = np.zeros(self.configuration.n_episodes)
        self.iterations_per_episode = []
        self.total_distortions = []

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

        self.agent.share_memory()

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

        process_handler = TorchProcsHandler(n_procs=self.configuration.n_procs)

        for p in range(self.configuration.n_procs):
            # worker_idx: int, worker_model: nn.Module, params: dir
            process_handler.create_process_and_start(target=worker,
                                                     args=(p, self.agent,
                                                           {"optimizer_type": self.configuration.optimizer_type,
                                                            "learning_rate": self.configuration.optimizer_type}))

        process_handler.join_and_terminate()



