"""Module pytorch_optimizer_configuration. Specifies a
data class for configuring PyTorch optimizers

"""

from dataclasses import dataclass
from src.maths.optimizer_type import OptimizerType


@dataclass(init=True, repr=True)
class PyTorchOptimizerConfig(object):
    """Configuration class for the optimizer

    """
    optimizer_type: OptimizerType = OptimizerType.ADAM
    optimizer_learning_rate: float = 0.01
    optimizer_eps = 1.0e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_weight_decay: float = 0
    optimizer_amsgrad: bool = False
    optimizer_maximize = False

    def as_dict(self) -> dict:
        return {"optimizer_type": self.optimizer_type,
                "learning_rate": self.optimizer_learning_rate,
                "eps": self.optimizer_eps,
                "betas": self.optimizer_betas,
                "weight_decay": self.optimizer_weight_decay,
                "amsgrad": self.optimizer_amsgrad,
                "maximize": self.optimizer_maximize}