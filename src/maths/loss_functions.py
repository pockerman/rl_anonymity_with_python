"""Module loss_functions. Implements basic loss functions
geared towards using PyTorch

"""

import torch


def mse(returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Mean square error loss function

    Parameters
    ----------
    returns: Values 1
    values: Values 2

    Returns
    -------
    A torch tensor representing the MSE loss
    """

    value_error = returns - values
    loss = value_error.pow(2).mul(0.5).mean()
    return loss
