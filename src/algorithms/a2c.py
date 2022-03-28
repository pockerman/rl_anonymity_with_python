import numpy as np
from typing import TypeVar, Generic, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


from src.utils.experience_buffer import unpack_batch
from src.utils.episode_info import EpisodeInfo
from src.utils.function_wraps import time_func_wrapper
from src.utils.replay_buffer import ReplayBuffer
from src.spaces.time_step import VectorTimeStep
from src.maths.pytorch_optimizer_config import PyTorchOptimizerConfig
from src.maths.pytorch_optimizer_builder import pytorch_optimizer_builder

Env = TypeVar("Env")
Optimizer = TypeVar("Optimizer")
LossFunction = TypeVar("LossFunction")
State = TypeVar("State")
Action = TypeVar("Action")
TimeStep = TypeVar("TimeStep")
Criteria = TypeVar('Criteria')


class A2CNetBase(nn.Module):
    """Base class for A2C networks

    """

    def __init__(self, architecture):
        super(A2CNetBase, self).__init__()
        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)


class A2CNet(nn.Module):

    def __init__(self, common_net: A2CNetBase, policy_net: A2CNetBase, value_net: A2CNetBase):
        super(A2CNet, self).__init__()
        self.common_net = common_net
        self.policy_net = policy_net
        self.value_net = value_net

    def forward(self, x):
        x = self.common_net(x)

        pol_out = self.policy_net(x)
        val_out = self.value_net(x)
        return pol_out, val_out


@dataclass(init=True, repr=True)
class A2CConfig(object):
    """Configuration for A2C algorithm

    """

    gamma: float = 0.99
    tau: float = 1.2
    n_iterations_per_episode: int = 100
    action_sampler: Callable = None
    loss_function: LossFunction = None
    batch_size: int = 0
    device: str = 'cpu'
    a2cnet: nn.Module = None
    save_model_path: Path = None
    optimizer_config: PyTorchOptimizerConfig = None


class A2C(Generic[Optimizer]):

    @staticmethod
    def update_parameters(optimizer: Optimizer, episode_info: EpisodeInfo, *, config: A2CConfig):
        """Update the parameters

        Parameters
        ----------

        optimizer: The optimizer instance used in training
        episode_info: The episode info
        config: The training configuration

        Returns
        -------

        """

        # unroll the batch notice the flip we go in reverse
        rewards = rewards = torch.Tensor(episode_info.info["replay_buffer"].to_numpy("reward")).flip(dims=(0,)).view(-1)
        logprobs = torch.stack(episode_info.info["logprobs"]).flip(dims=(0,)).view(-1)
        values = torch.stack(episode_info.info["values"]).flip(dims=(0,)).view(-1)

        returns = []
        ret_ = torch.Tensor([0])

        # Loop through the rewards in reverse order to generate
        # R = r i + γ * R
        for r in range(rewards.shape[0]):  # B
            ret_ = rewards[r] + config.gamma * ret_
            returns.append(ret_)

        returns = torch.stack(returns).view(-1)
        returns = F.normalize(returns, dim=0)

        # compute the actor loss.
        # Minimize the actor loss: –1 * γ t * (R – v(s t )) * π (a ⏐ s)
        actor_loss = -1 * logprobs * (returns - values.detach())  # C

        # compute the critic loss. Minimize the critic loss: (R – v) 2 .
        critic_loss = torch.pow(values - returns, 2)  # D
        loss = actor_loss.sum() + config.tau * critic_loss.sum()  # E
        loss.backward()
        optimizer.step()

    @classmethod
    def from_path(cls, config: A2CConfig, path: Path):
        """Load the A2C model parameters from the given path

        Parameters
        ----------
        config: The configuration of the algorithm
        path: The path to load the parameters

        Returns
        -------

        An instance of A2C class
        """
        a2c = A2C(config)
        a2c.config.a2cnet.load_state_dict(torch.load(path))
        a2c.set_test_mode()
        return a2c

    def __init__(self, config: A2CConfig):

        self.config: A2CConfig = config
        self.optimizer: Optimizer = None
        self.name = "A2C"

    @property
    def a2c_net(self) -> nn.Module:
        return self.config.a2cnet

    def __call__(self, x: torch.Tensor):
        return self.a2c_net(x)

    def share_memory(self) -> None:
        """Instruct the underlying network to
        set up what is needed to share memory

        Returns
        -------

        None
        """
        self.a2c_net.share_memory()

    def parameters(self) -> Any:
        """The parameters of the underlying model

        Returns
        -------

        """
        return self.a2c_net.parameters()

    def set_train_mode(self) -> None:
        """Set the model to a training mode

        Returns
        -------
        None
        """
        self.a2c_net.train()

    def set_test_mode(self) -> None:
        """Set the model to a testing mode

        Returns
        -------
        None
        """
        self.a2c_net.eval()

    def save_model(self, path: Path) -> None:
        """Save the model on a file at the given path

        Parameters
        ----------
        path: The path to save the model

        Returns
        -------

        None
        """
        torch.save(self.a2c_net.state_dict(), Path(str(path) + "/" + self.name + ".pt"))

    def play(self, env: Env, criteria: Criteria):
        """Play the agent on the environment

        Parameters
        ----------
        env: The environment to test/play the agent
        criteria: The criteria to stop the game

        Returns
        -------

        """

        time_step = env.reset()

        while criteria.continue_itrs():
            state = time_step.observation.to_numpy()
            state = torch.from_numpy(state).float()
            logits, values = self(state)

            # select action
            action = None
            time_step = env.step(action)

            if time_step.done:
                time_step = env.reset()

    def optimize_model(self, logpas, entropies, values, rewards, n_workers) -> None:
        logpas = torch.stack(logpas).squeeze()
        entropies = torch.stack(entropies).squeeze()
        values = torch.stack(values).squeeze()

        T = len(rewards)
        discounts = np.logspace(0, T, num=T, base=self.config.gamma, endpoint=False)
        rewards = np.array(rewards).squeeze()
        returns = np.array([[np.sum(discounts[:T - t] * rewards[t:, w]) for t in range(T)]
                            for w in range(n_workers)])

        np_values = values.data.numpy()
        tau_discounts = np.logspace(0, T - 1, num=T - 1, base=self.config.gamma * self.config.tau, endpoint=False)
        advs = rewards[:-1] + self.config.gamma * np_values[1:] - np_values[:-1]
        gaes = np.array([[np.sum(tau_discounts[:T - 1 - t] * advs[t:, w]) for t in range(T - 1)]
                         for w in range(n_workers)])
        discounted_gaes = discounts[:-1] * gaes

        values = values[:-1, ...].view(-1).unsqueeze(1)
        logpas = logpas.view(-1).unsqueeze(1)
        entropies = entropies.view(-1).unsqueeze(1)
        returns = torch.FloatTensor(returns.T[:-1]).view(-1).unsqueeze(1)
        discounted_gaes = torch.FloatTensor(discounted_gaes.T).view(-1).unsqueeze(1)

        T -= 1
        T *= n_workers

        assert returns.size() == (T, 1)
        assert values.size() == (T, 1)
        assert logpas.size() == (T, 1)
        assert entropies.size() == (T, 1)

        value_error = returns.detach() - values
        value_loss = value_error.pow(2).mul(0.5).mean()
        policy_loss = -(discounted_gaes.detach() * logpas).mean()
        entropy_loss = -entropies.mean()
        loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss + self.entropy_loss_weight * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.ac_model_max_grad_norm)
        self.optimizer.step()

    def actions_before_training_begins(self, env: Env, **options) -> None:

        # build the optimizer we need in order to train the model
        self.optimizer = pytorch_optimizer_builder(opt_type=self.config.optimizer_config.optimizer_type,
                                                   model_params=self.parameters(),
                                                   **self.config.optimizer_config.as_dict())

    def actions_before_episode_begins(self, env: Env, episode_idx: int, **options) -> None:
        self.set_train_mode()
        self.optimizer.zero_grad()

    def actions_after_training(self) -> None:
        """Any actions the agent needs to perform after training

        Returns
        -------

        """

        if self.config.save_model_path is not None:
            self.save_model(path=self.config.save_model_path)

    def actions_after_episode_ends(self, env: Env, episode_idx: int, **options) -> None:

        self.optimize_model(env.n_workers())

    def on_episode(self, env: Env, episode_idx: int,  **options) -> EpisodeInfo:
        """Train the algorithm on the episode

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        episode_info, total_time = self._do_train(env, episode_idx, **options)
        episode_info.total_execution_time = total_time
        return episode_info

    @time_func_wrapper(show_time=False)
    def _do_train(self, env: Env, episode_idx: int, **options) -> EpisodeInfo:
        """Train the algorithm on the episode. In fact this method simply
        plays the environment to collect batches

        Parameters
        ----------

        env: The environment to train on
        episode_idx: The index of the training episode
        options: Any keyword based options passed by the client code

        Returns
        -------

        An instance of EpisodeInfo
        """

        # episode score
        episode_score = 0
        episode_iterations = 0
        total_distortion = 0

        buffer = ReplayBuffer(options["buffer_size"])

        time_step: VectorTimeStep = env.reset()
        states = time_step.stack_observations() #torch.from_numpy(time_step.observation.to_numpy()).float()

        # represent the probabilities under the
        # policy
        logprobs = []
        entropies = []
        rewards = []
        values = []
        for itr in range(self.config.n_iterations_per_episode):

            # make a full pass on the model
            action, is_exploratory, logprob, entropy, value = self._full_pass(state=states)

            # step with the given actions
            time_step: VectorTimeStep = env.step(action)

            episode_score += time_step.reward
            total_distortion += time_step.info["total_distortion"]

            state = time_step.stack_observations()

            buffer.add(state=state, action=action, reward=time_step.reward,
                       next_state=time_step.observation, done=time_step.done)

            if time_step.done:
                break

            episode_iterations += 1

        episode_info = EpisodeInfo(episode_score=episode_score,
                                   total_distortion=total_distortion, episode_itrs=episode_iterations,
                                   info={"replay_buffer": buffer,
                                         "logprobs": logprobs,
                                         "values": values})
        return episode_info


    def _full_pass(self, state) -> tuple:

        # policy and critic values
        policy, value = self.a2c_net(state)

        logits = policy.view(-1)

        ##dist = torch.distributions.Categorical(logits=logits)
        ##action = dist.sample()

        # choose the action. Typically this will be Categorical
        # but we leave it open for the application
        action_sampler_dist = self.config.action_sampler(logits)
        action = action_sampler_dist.sample()

        # the log probabilities of the policy
        logprob = action_sampler_dist.log_prob(action).unsqueeze(-1)#policy.view(-1)[action]
        entropy = action_sampler_dist.entropy().unsqueeze(-1)
        #logprobs.append(logprob_)

        #logpa = dist.log_prob(action).unsqueeze(-1)
        #entropy = dist.entropy().unsqueeze(-1)

        action = action.item() if len(action) == 1 else action.data.numpy()
        is_exploratory = action != np.argmax(logits.detach().numpy(), axis=int(len(state) != 1))
        return action, is_exploratory, logprob, entropy, value

    def _interaction_step(self, states, env: Env):
        actions, is_exploratory, logpas, entropies, values = self.ac_model.full_pass(states)
        new_states, rewards, is_terminals, _ = env.step(actions)

        self.logpas.append(logpas);
        self.entropies.append(entropies)
        self.rewards.append(rewards);
        self.values.append(values)

        self.running_reward += rewards
        self.running_timestep += 1
        self.running_exploration += is_exploratory[:, np.newaxis].astype(np.int)

        return new_states, is_terminals