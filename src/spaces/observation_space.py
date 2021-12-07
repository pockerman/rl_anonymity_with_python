
import gym


class ObsSpace(gym.spaces.Space):

    def __init__(self, ds):
        super(ObsSpace, self).__init__(shape=ds.shape, dtype=None)
        self.ds = ds
        self.dtype = "PandasDF"

    def contains(self, obs):
        return True