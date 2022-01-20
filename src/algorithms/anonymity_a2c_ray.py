import copy
import gym
import pandas as pd
from gym.spaces import Discrete
import ray
import ray.rllib.agents.a3c as a3c
from ray.tune.logger import pretty_print
from ray.rllib.env.env_context import EnvContext
from src.spaces.discrete_state_environment import TimeStep, StepType
from src.spaces.observation_space import ObsSpace


class DataSetEnv(gym.Env):


    def __init__(self, env_config: EnvContext):
        super(DataSetEnv, self).__init__()
        self.gamma = env_config["gamma"]
        self.ds = copy.deepcopy(env_config["ds"])
        self.start_ds = copy.deepcopy(env_config["ds"])
        self.action_space = Discrete(2)
        self.observation_space = ObsSpace(ds=self.ds) #Box(0.0, 0, shape=(1,), dtype=np.float32)
        # Set the seed. This is only used for the final (reach goal) reward.
        self.seed(env_config.worker_index * env_config.num_workers)
        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0, discount=self.gamma, observation=self.start_ds)


    def reset(self) -> TimeStep:
        """
        Starts a new sequence and returns the first `TimeStep` of this sequence.
        Returns:
          A `TimeStep` namedtuple containing:
            step_type: A `StepType` of `FIRST`.
            reward: `None`, indicating the reward is undefined.
            discount: `None`, indicating the discount is undefined.
            observation: A NumPy array, or a nested dict, list or tuple of arrays.
              Scalar values that can be cast to NumPy arrays (e.g. Python floats)
              are also valid in place of a scalar array. Must conform to the
              specification returned by `observation_spec()`.
        """
        self.current_time_step = TimeStep(step_type=StepType.FIRST, reward=0.0, discount=self.gamma,
                                          observation=self.start_ds)
        return self.current_time_step.observation.to_numpy()

    def step(self, action) -> TimeStep:
        """

        Updates the environment according to the action and returns a `TimeStep`.
        If the environment returned a `TimeStep` with `StepType.LAST` at the
        previous step, this call to `step` will start a new sequence and `action`
        will be ignored.

        This method will also start a new sequence if called after the environment
        has been constructed and `reset` has not been called. Again, in this case
        `action` will be ignored.
        """
        pass

    def render(self, mode="human"):
        pass

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass


if __name__ == '__main__':

    data = pd.DataFrame([(34, "male", 81667), (45, "female", 81675)],
                        columns=["Age", "Gender", "ZipCode"])

    print(data.head())

    ray.init(local_mode=True)
    config = a3c.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 1

    config["env_config"] = {"ds": data, "gamma": 1.0}

    #env = DataSetEnv(ds=data, env_config=None)
    #env = DataSetEnv(ds=data, env_config=None)
    trainer = a3c.A2CTrainer(config=config, env=DataSetEnv)

    for i in range(10):
        print("Training agent...")
        result = trainer.train()
        print(pretty_print(result))

