A2C algorithm on three columns data set
=======================================


A2C algorithm
==============

Both the Q-learning algorithm we used in `Q-learning on a three columns dataset <qlearning_three_columns.html>`_ and the SARSA algorithm in 
`Semi-gradient SARSA on a three columns data set`_ are value-based methods; that is they estimate value functions. Specifically the state-action function
:math:`Q`. By knowing :math:`Q` we can construct a policy to follow for example to choose the action that at the given state
maximizes the state-action function i.e. :math:`argmax_{\alpha}Q(s_t, \alpha)` i.e. a greedy policy. 

However, the true objective of reinforcement learning is to directly learn a policy  :math:`\pi`.


The main advantage of learning a parametrized policy is that it can be any learnable function e.g. a linear model or a deep neural network.

The A2C algorithm falls under the umbrella of actor-critic methods [REF]. In these methods, we estimate a parametrized policy; the actor
and a parametrized value function; the critic.


Specifically, we will use a weight-sharing model. Moreover, the environment is a multi-process class that gathers samples from multiple
emvironments at once
