import unittest
import pytest

from src.spaces import MultiprocessEnv, TimeStep, StepType

class DummyEnv(object):

    def __init__(self, **options):
        pass

    def close(self, **kwargs):
        pass

    def step(self, **kwargs) -> TimeStep:
        print("Action executed={0}".format(kwargs["action"]))
        time_step = TimeStep(step_type=StepType.FIRST if kwargs["action"] == 1 else StepType.LAST,
                             reward=0.0, observation=1.0, info={}, discount=0.0)
        return time_step

    def reset(self, **kwargs) -> TimeStep:
        time_step = TimeStep(step_type=StepType.FIRST,
                             reward=0.0, observation=1.0, info={}, discount=0.0)
        return time_step


class TestMultiprocessEnv(unittest.TestCase):

    @staticmethod
    def make_environment(options):
        return DummyEnv(**options)

    def test_make(self):
        options = {}
        multiproc_env = MultiprocessEnv(TestMultiprocessEnv.make_environment, options, n_workers=2)

        try:
            multiproc_env.make()
            multiproc_env.close()
        except Exception as e:
            print("Test failed due to excpetion {0}".format(str(e)))
            multiproc_env.close()

    def test_step_fail(self):

        options = {}
        multiproc_env = MultiprocessEnv(TestMultiprocessEnv.make_environment, options, n_workers=2)

        with pytest.raises(ValueError) as e:
            multiproc_env.make()
            multiproc_env.step([])

        multiproc_env.close()
        self.assertEqual("Number of actions is not equal to the number of workers", str(e.value))

    def test_step(self):

        options = {}
        multiproc_env = MultiprocessEnv(TestMultiprocessEnv.make_environment, options, n_workers=2)

        multiproc_env.make()
        time_step = multiproc_env.step([1, 2])
        multiproc_env.close()
        self.assertEqual(len(multiproc_env), len(time_step))
        self.assertEqual(StepType.FIRST, time_step[0].step_type)
        self.assertEqual(StepType.LAST, time_step[1].step_type)

    def test_reset(self):
        options = {}
        multiproc_env = MultiprocessEnv(TestMultiprocessEnv.make_environment, options, n_workers=2)

        multiproc_env.make()
        time_step = multiproc_env.reset()
        multiproc_env.close()
        self.assertEqual(len(multiproc_env), len(time_step))


if __name__ == '__main__':
    unittest.main()