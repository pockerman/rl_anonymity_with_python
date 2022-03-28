from typing import Callable

import torch.multiprocessing as mp


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

    def __len__(self) -> int:
        """The number of workers handled by this
        instance

        Returns
        -------

        """
        return len(self.processes)

    def create_and_start(self, target: Callable, *args) -> None:
        for i in range(self.n_procs):
            p = mp.Process(target=target, args=args)
            p.start()
            self.processes.append(p)

    def create_process_and_start(self, target: Callable, args) -> None:
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
