"""
Linear action-value (q-value) function approximator for
semi-gradient methods with state-action featurization via tile coding.
See the implementation of the n-step semi-gradient SARSA algorithm in sarsa_semi_gradient module.
Initial implementation is taken from https://michaeloneill.github.io/RL-tutorial.html
"""

import numpy as np
from typing import TypeVar

State = TypeVar('State')
Action = TypeVar('Action')
Env = TypeVar('Env')


class QEstimator(object):
    def __init__(self, env: Env, max_size: int, alpha: float, use_trace: bool = False) -> None:
        self.env = env
        self.max_size = max_size
        self.alpha = alpha
        self.use_trace = use_trace
        self.weights = np.zeros(max_size)

        if self.use_trace:
            self.z = np.zeros(max_size)

    def predict(self, state: State, action: Action = None) -> list:

        """
        Predicts q-value(s) using linear FA.
        If action a is given then returns prediction
        for single state-action pair (s, a).
        Otherwise returns predictions for all actions
        in environment paired with s.
        :param state:
        :param action:
        :return:
        """

        if action is None:
            features = [self.env.featurize_state_action(state, i) for i in range(self.env.n_actions)]
        else:
            features = [self.env.featurize_state_action(state, action)]

        return [np.sum(self.weights[f]) for f in features]

    def update(self, state: State, action: Action, target):
        """
        Updates the estimator parameters
        for a given state and action towards
        the target using the gradient update rule
        (and the eligibility trace if one has been set).
        """
        features = self.env.featurize_state_action(state, action)

        # Linear FA
        estimation = np.sum(self.weights[features])
        delta = (target - estimation)

        if self.use_trace:
            # self.z[features] += 1  # Accumulating trace
            self.z[features] = 1  # Replacing trace
            self.weights += self.alpha * delta * self.z
        else:
            self.weights[features] += self.alpha * delta

    def reset(self, z_only: bool = False):
        """
        Resets the eligibility trace (must be done at
        the start of every epoch) and optionally the
        weight vector (if we want to restart training
        from scratch).
        """

        if z_only:
            assert self.use_trace, 'q-value estimator has no z to reset.'
            self.z = np.zeros(self.max_size)
        else:
            if self.use_trace:
                self.z = np.zeros(self.max_size)
            self.weights = np.zeros(self.max_size)
