A2C algorithm on mock data set
==============================

Overview
--------

Both the Q-learning algorithm we used in `Q-learning on a three columns dataset <qlearning_three_columns.html>`_ and the SARSA algorithm in 
`Semi-gradient SARSA on a three columns data set <semi_gradient_sarsa_three_columns.html>`_ are value-based methods; that is they estimate directly value functions. Specifically the state-action function
:math:`Q`. By knowing :math:`Q` we can construct a policy to follow for example to choose the action that at the given state
maximizes the state-action function i.e. :math:`argmax_{\alpha}Q(s_t, \alpha)` i.e. a greedy policy. These methods are called off-policy methods. 

However, the true objective of reinforcement learning is to directly learn a policy  :math:`\pi`. One class of algorithms towards this directions are policy gradient algorithms
like REINFORCE and Advantage Actor-Critic or A2C algorithms. A review of A2C methods can be found in [1]


A2C algorithm
-------------

Typically with policy gradient methods and A2C in particular, we approximate directly the policy by a parametrized model.
Thereafter, we train the model i.e. learn its paramters by taking samples from the environment. 
The main advantage of learning a parametrized policy is that it can be any learnable function e.g. a linear model or a deep neural network.

The A2C algorithm  is a the synchronous version of A3C [REF]. Both algorithms, fall under the umbrella of actor-critic methods [REF]. In these methods, we estimate a parametrized policy; the actor
and a parametrized value function; the critic. The role of the policy or actor network is to indicate which action to take on a given state. In our implementation below,
the policy network returns a probability distribution over the action space. Specifically,  a tensor of probabilities. The role of the critic model is to evaluate how good is
the action that is selected.

In A2C there is a single agent that interacts with multiple instances of the environment. In other words, we create a number of workers where each worker loads its own instance of the data set to anonymize. A shared model is then optimized by each worker.

The objective of the agent is to maximize the expected discounted return [2]: 

.. math:: 

   J(\pi_{\theta}) = E_{\tau \sim \rho_{\theta}}\left[\sum_{t=0}^T\gamma^t R(s_t, \alpha_t)\right]
   
where :math:`\tau` is the trajectory the agent observes with probability distribution :math:`\rho_{\theta}`, :math:`\gamma` is the 
discount factor and :math:`R(s_t, \alpha_t)` represents some unknown to the agent reward function. We can rewrite the expression above as

.. math:: 

   J(\pi_{\theta}) = E_{\tau \sim \rho_{\theta}}\left[\sum_{t=0}^T\gamma^t R(s_t, \alpha_t)\right] = \int \rho_{\theta} (\tau) \sum_{t=0}^T\gamma^t R(s_t, \alpha_t) d\tau


Let's condense the involved notation by using :math:`G(\tau)` to denote the sum in the expression above i.e.

.. math::

   G(\tau) = \sum_{t=0}^T\gamma^t R(s_t, \alpha_t)
   
The probability distribution :math:`\rho_{\theta}` should be a function of the followed policy :math:`\pi_{\theta}` as this dictates what action is followed. Indeed we can write  [2],

.. math::

   \rho_{\theta} = p(s_0) \Pi_{t=0}^{\infty} \pi_{\theta}(a_t, s_t)P(s_{t+1}| s_t, a_t)

  
where :math:` P(s_{t+1}| s_t, a_t)` denote state transition probabilities. 
Policy gradient methods use the gardient of :math:`J(\pi_{\theta})` in order to make progress. It turns out, see for example [2, 3] that we can write

.. math::

   \nabla_{\theta} J(\pi_{\theta}) = \int \rho_{\theta}  \nabla_{\theta} log (\rho_{\theta})  G(\tau) d\tau

However, we cannot fully evaluate the integral above as we don't know the transition probabilities. Thus, we resort into taking samples from the
environment in order to obtain an estimate.


Specifically, we will use a weight-sharing model. Moreover, the environment is a multi-process class that gathers samples from multiple
emvironments at once.

The advanatge :math:`A(s_t, \alpha_t)` is defined as [REF]

.. math::
	
	A(s_t, \alpha_t) = Q_{\pi}(s_t, \alpha_t) - V_{\pi}(s_t)
	
It represents a goodness fit for an action at a given state. We can use the critic function for :math:`V_{\pi}(s_t)` and use the following approximation
for :math:`Q_{\pi}(s_t, \alpha_t)`

.. math::

   Q_{\pi}(s_t, \alpha_t) = r_t + \gamma V_{\pi}(s_{t+1})
   
leading to 


.. math::
	
	A(s_t, \alpha_t) = r_t + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t) 



The critic loss function is 

So the gradient becomes

.. math::

   \nabla_{\theta}J(\theta) \approx \sum_{t=0}^{T-1} \nabla_{\theta}log \pi_{\theta}(\alpha_t | s_t)A(s_t, \alpha_t)

 

Overall, the A2C algorithm is described below

1. Initialize the network parameters $\theta$
2. Play :math:`N` steps in the environment using the current policy :math:`\pi_{\theta}`
3. Loop over the accumulated experience in reversed order :math:`T, t-1, \dots, t_0`
	- Copute the total reward :math:`R = r_t + \gamma R`
	- Compute Actor gradients
	- Compute Critic gradients
Code
----


References
----------

1. Ivo Grondman, Lucian Busoniu, Gabriel A. D. Lopes, Robert Babuska, A survey of Actor-Critic reinforcement learning: Standard and natural policy gradients. IEEE Transactions on Systems, Man and Cybernetics-Part C Applications and Reviews, vol 12, 2012.
2. Enes Bilgin, Mastering reinforcement learning with python. Packt Publishing.
3. Miguel Morales, Grokking deep reinforcement learning. Manning Publications.
