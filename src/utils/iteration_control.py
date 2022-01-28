"""
Utility to control iteration
"""

from src.utils import INFO, VERSION


class IterationControl(object):
    """
    Helper class to control iteration
    """

    def __init__(self, n_itrs: int, min_dist: float, max_dist: float) -> None:
        self.n_itrs = n_itrs
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.iteration_counter = 0

    def continue_itr(self, distortion: float):

        if self.min_dist <= distortion <= self.max_dist:
            print("{0} Finished iteration with distortion={1} "
                  "in [{2}, {3}]. Number of iterations={4}".format(INFO, distortion,
                                                                   self.min_dist, self.max_dist,
                                                                   self.iteration_counter))
            return False

        # TODO: There is no way currently we go back
        # to acceptable distortions
        if VERSION == '0.0.3-alpha':
            if distortion > self.max_dist:
                print("{0} Finished iteration with distortion={1} "
                      "in [{2}, {3}]. Number of iterations={4}".format(INFO, distortion, self.min_dist,
                                                                       self.max_dist, self.iteration_counter))
                return False

        if 0 <= self.n_itrs <= self.iteration_counter:
            print("{0} Reached maximum number of iterations. With distortion={1} "
                  "in [{2}, {3}]. Number of iterations={4}".format(INFO, distortion,
                                                                   self.min_dist, self.max_dist,
                                                                   self.iteration_counter))
            return False

        self.iteration_counter += 1
        print("{0} Iteration={1} Distortion={2}".format(INFO, self.iteration_counter, distortion))
        return True
