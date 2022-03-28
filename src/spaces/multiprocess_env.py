"""Module multiprocess_env. Specifies
a vectorized environment where each instance
of the environment is run independently. The implementation
of the environment is taken from the book
Grokking Deep Reinforcement Learning Algorithms
by Manning publications

"""

import numpy as np
from typing import TypeVar, Callable, Any
import torch.multiprocessing as mp

from src.spaces import TimeStep, VectorTimeStep
from src.parallel import TorchProcsHandler


ActionVector = TypeVar('ActionVector')


class MultiprocessEnv(object):

    def __init__(self, env_builder: Callable, env_args: dict, n_workers: int):
        self.env_builder = env_builder
        self.env_args = env_args
        self.n_workers = n_workers
        self.workers = TorchProcsHandler(n_procs=n_workers)
        self.pipes = [mp.Pipe() for _ in range(self.n_workers)]

    def __len__(self) -> int:
        """The number of workers handled by this
        instance

        Returns
        -------

        """
        return len(self.workers)

    def make(self):
        """Create the workers

        Returns
        -------

        """

        for w in range(self.n_workers):
            self.workers.create_process_and_start(target=self.work, args=(w, self.env_builder,
                                                                          self.env_args,
                                                                          self.pipes[w][1]))

    def work(self, rank, env_builder: Callable, env_args: dict, pipe_end) -> None:
        """The worker function

        Parameters
        ----------
        rank: The rank of the worker
        env_builder: The callable that builds the worker environment
        env_args: The callable arguments
        worker_end

        Returns
        -------
        None
        """

        # create the environment
        env = env_builder(env_args)
        while True:

            # receive new cmd from the manager
            # in order to exceute it
            cmd, kwargs = pipe_end.recv()

            if cmd == 'reset':
                pipe_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                pipe_end.send(env.step(**kwargs))
            elif cmd == '_past_limit':
                pipe_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                # including close command
                env.close(**kwargs)
                del env
                pipe_end.close()
                break

    def reset(self, rank=None, **kwargs) -> VectorTimeStep:

        time_step = VectorTimeStep()
        if rank is not None:
            parent_end, _ = self.pipes[rank]
            self._send_msg(('reset', {}), rank)
            o = parent_end.recv()
            time_step.append(0)
            return time_step

        # if not reset for  a specific worker
        # then all workers should reset
        self._broadcast_msg(('reset', kwargs))

        # collect all the timesteps from the
        # workers
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            process_time_step = parent_end.recv()
            time_step.append(process_time_step)

        return time_step

    def step(self, actions: ActionVector) -> VectorTimeStep:

        if len(actions) != self.n_workers:
            raise ValueError("Number of actions is not equal to the number of workers")

        # send the messages to the workers
        [self._send_msg(('step', {'action': actions[rank]}), rank) for rank in range(self.n_workers)]

        time_step = VectorTimeStep()
        # collect the results from all processes
        #results = []

        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            process_time_step = parent_end.recv()
            time_step.append(process_time_step)
            """
            o, r, d, i = parent_end.recv()
            results.append((o,
                            np.array(r, dtype=np.float),
                            np.array(d, dtype=np.float),
                            i))
        return [np.vstack(block) for block in np.array(results).T]
        """
        return time_step

    def close(self, **kwargs):
        self._close(**kwargs)

    def _close(self, **kwargs):
        self._broadcast_msg(('close', kwargs))

    def _send_msg(self, msg: Any, rank: int):
        """Send the message to the process with the
        given rank

        Parameters
        ----------
        msg: The message to send
        rank: The rank of the proces to send the message

        Returns
        -------

        """
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def _broadcast_msg(self, msg):
        """Broadcast the message to all workers

        Parameters
        ----------
        msg

        Returns
        -------

        """
        [parent_end.send(msg) for parent_end, _ in self.pipes]
