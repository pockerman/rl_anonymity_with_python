"""Module epsilon_greedy_q_estimator

"""

from src.utils.mixins import WithEstimatorMixin

class EpsilonGreedyQEstimator(WithEstimatorMixin):
    
    def __init__(self):
        super(EpsilonGreedyQEstimator, self).__init__()