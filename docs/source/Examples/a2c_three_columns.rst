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

The A2C algorithm  is a the synchronous version of A3C [2]. Both algorithms, fall under the umbrella of actor-critic methods. In these methods, we estimate a parametrized policy; the actor
and a parametrized value function; the critic. The role of the policy or actor network is to indicate which action to take on a given state. In our implementation below,
the policy network returns a probability distribution over the action space. Specifically,  a tensor of probabilities. The role of the critic model is to evaluate how good is
the action that is selected.

In A2C there is a single agent that interacts with multiple instances of the environment. In other words, we create a number of workers where each worker loads its own instance of the data set to anonymize. A shared model is then optimized by each worker.

The objective of the agent is to maximize the expected discounted return [2]: 

.. math:: 

   J(\pi_{\theta}) = E_{\tau \sim \rho_{\theta}}\left[\sum_{t=0}^T\gamma^t R(s_t, a_t)\right]
   
where :math:`\tau` is the trajectory the agent observes with probability distribution :math:`\rho_{\theta}`, :math:`\gamma` is the 
discount factor and :math:`R(s_t, \alpha_t)` represents some unknown to the agent reward function. We can rewrite the expression above as

.. math:: 

   J(\pi_{\theta}) = E_{\tau \sim \rho_{\theta}}\left[\sum_{t=0}^T\gamma^t R(s_t, a_t)\right] = \int \rho_{\theta} (\tau) \sum_{t=0}^T\gamma^t R(s_t, a_t) d\tau


Let's condense the involved notation by using :math:`G(\tau)` to denote the sum in the expression above i.e.

.. math::

   G(\tau) = \sum_{t=0}^T\gamma^t R(s_t, a_t)
   
The probability distribution :math:`\rho_{\theta}` should be a function of the followed policy :math:`\pi_{\theta}` as this dictates what action is followed. Indeed we can write  [2],

.. math::

   \rho_{\theta} = p(s_0) \Pi_{t=0}^{\infty} \pi_{\theta}(a_t, s_t)P(s_{t+1}| s_t, a_t)

  
where :math:`P(s_{t+1}| s_t, a_t)` denotes the state transition probabilities. 
Policy gradient methods use the gardient of :math:`J(\pi_{\theta})` in order to make progress. It turns out, see for example [2, 3] that we can write

.. math::

   \nabla_{\theta} J(\pi_{\theta}) = \int \rho_{\theta}  \nabla_{\theta} log (\rho_{\theta})  G(\tau) d\tau

This equation above forms the essence of the policy gradient methods. However, we cannot fully evaluate the integral above as we don't know the transition probabilities.  We can eliminate the 
term that involves the gradient :math:`\nabla_{\theta}\rho_{\theta}` by using the expression for :math:`\rho_{\theta}`

.. math::
   
   \nabla_{\theta}log(\rho_{\theta}) = \nabla_{\theta}log\left[p(s_0) \Pi_{t=0}^{\infty} \pi_{\theta}(a_t, s_t)P(s_{t+1}| s_t, a_t)\right]

From the expression above only the term :math:`\pi_{\theta}(a_t, s_t)` involves :math:`\theta`. Thus,

.. math::
 
   \nabla_{\theta}log(\rho_{\theta}) = \sum_{t=0}^{\infty} \nabla_{\theta}log(\pi_{\theta}(a_t, s_t)


We will use the expression above as well as batches of trajectories in order to calculate the integral above. In particular,
we will use the following expression

.. math::

   \nabla_{\theta} J(\pi_{\theta}) \approx \frac{1}{N}\sum_{i=1}^{N}\left( \sum_{t=0}^{T} \nabla_{\theta}log(\pi_{\theta}(a_t, s_t) \right) G(\tau)
   
where :math:`N` is the size of the batch. There are various expressions for :math:`G(\tau)` (see e.g. [4]) . Belowe, we review some of them. 
The first expression is given by 

.. math::

   G(\tau) = \sum_{t=0}^T\gamma^t R(s_t, a_t)
   
and this is the expression used by the REINFORCE algorithm [2].  However, this is a full Monte Carlo estimate and  when :math:`N` is small the gradient estimation may exhibit high variance. In such cases learning may not be stable.  Another expression we could employ is known as the reward-to-go term [2]:

.. math::

   G(\tau) = \sum_{t^{'} = t}^T\gamma^t R(s_{t^{`}}, a_{t^{`}})
   
Another idea is to use a baseline in order to reduce further the gradient variance [2]. One such approach is to use the so-called advantage function :math:`A(s_t, \alpha_t)` defined  as [2]

.. math::
	
	A(s_t, a_t) = Q_{\pi}(s_t, a_t) - V_{\pi}(s_t)
	
	
The advantage function measures how much the agent is better off by taking action :math:`a_t` when in state :math:`s_t` as opposed to following the existing policy. 
Let's see how we can estimate the advantage function.


Estimate :math:`A(s_t, a_t)`
----------------------------

The advantage function involes both the state-action value function :math:`Q_{\pi}(s_t, a_t)` as well as the value function :math:`V_{\pi}(s_t)`.
Given a model that somehow estimates :math:`V_{\pi}(s_t)`, we can estimate  :math:`Q_{\pi}(s_t, a_t)` from

.. math::

   Q_{\pi}(s_t, a_t) \approx G(\tau)
   
or 

.. math::

   Q_{\pi}(s_t, a_t) \approx r_{t+1} + \gamma V_{\pi}(s_{t+1})
   
Resulting in 

.. math::

   A(s_t, a_t) = r_{t+1} + \gamma V_{\pi}(s_{t+1}) - V_{\pi}(s_t)



GAE 
----


The advantage actor-critic model we use in this section involves a more general form of the advanatge estimation known as Generalized Advanatge Estimation  or GAE.
This is a method for estimating targets for the advantage function [3]. Specifically, we use the following expression for the advantage function [4]


.. math::

   A(s_t, a_t)^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+1}


when :math:`\lambda=0` this expression results to the the expression for :math:`A(s_t, a_t)` [4].


A2C model
---------

As we already mentioned, in actor-critic methods, there are two models; the actor
and the critic. The role of the policy or actor model is to indicate which action to take on a given state
There are two main architectures for actor-critic methods; completely isolated actor and critic models or weight sharing models [2].
In the former, the two models share no common aspects. The advantage of such an approach is that it is usually more stable.
The second architecture allows for the two models to share some characteristics and differentiate in the last layers. Although this second option
requires careful tuning of the hyperparameters, it has the advantage of cross learning and use common extraction capabilities [2].

In this example, we will follow the second architecture. Moreover, to speed up training, we will use a multi-process environment
that gathers samples from multiple environments at once.

The loss function, we minimize is a weighted sum of the two loss functions of the participating models i.e.


.. math::

   L(\theta) = w_1 L_{\pi}(\theta) + w_2 L_{V_{\pi}}(\theta)

where

.. math::

   L_{\pi}(\theta) = J(\pi(\theta)) ~~  L_{V_{\pi}}(\theta) = MSE(y_i, V_{\pi}(s_i))


where :math:`MSE` is the mean square error function and :math:`y_i` are the state-value targets i.e.

.. math::

   y_i = r_i + \gamma V_{\pi}(s_{i}^{'}), ~~ i = 1, \cdots, N
   
   
Code
----

.. code-block::

	if __name__ == '__main__':
	    # set the seed for random engine
	    random.seed(42)

	    # set the seed for PyTorch
	    torch.manual_seed(42)

	    # this the A2C network
	    net = A2CNetSimpleLinear(n_columns=N_COLUMNS, n_actions=ACTION_SPACE_SIZE)

	    # agent configuration
	    a2c_config = A2CConfig(action_sampler=action_sampler, n_iterations_per_episode=N_ITRS_PER_EPISODE,
		                   a2cnet=net, save_model_path=Path("./a2c_three_columns_output/"),
		                   n_workers=N_WORKERS,
		                   optimizer_config=PyTorchOptimizerConfig(optimizer_type=OptimizerType.ADAM,
		                                                           optimizer_learning_rate=ALPHA))

	    # create the agent
	    agent = A2C(a2c_config)

	    # create a trainer to train the Qlearning agent
	    configuration = PyTorchTrainerConfig(n_episodes=N_EPISODES)

	    # set up the arguments
	    env = MultiprocessEnv(env_builder=env_loader, env_args={}, n_workers=N_WORKERS)

	    try:

		env.make(agent=agent)
		trainer = PyTorchTrainer(env=env, agent=agent, config=configuration)

		# train the agent
		trainer.train()

		avg_rewards = trainer.total_rewards
		plot_running_avg(avg_rewards, steps=100,
		                 xlabel="Episodes", ylabel="Reward",
		                 title="Running reward average over 100 episodes")

		avg_episode_dist = np.array(trainer.total_distortions)
		print("{0} Max/Min distortion {1}/{2}".format(INFO, np.max(avg_episode_dist), np.min(avg_episode_dist)))

		plot_running_avg(avg_episode_dist, steps=100,
		                 xlabel="Episodes", ylabel="Distortion",
		                 title="Running distortion average over 100 episodes")

	    except Exception as e:
		print("An excpetion was thrown...{0}".format(str(e)))
	    finally:
		env.close()

Results
--------

References
----------

1. Ivo Grondman, Lucian Busoniu, Gabriel A. D. Lopes, Robert Babuska, A survey of Actor-Critic reinforcement learning: Standard and natural policy gradients. IEEE Transactions on Systems, Man and Cybernetics-Part C Applications and Reviews, vol 12, 2012.
2. Enes Bilgin, Mastering reinforcement learning with python. Packt Publishing.
3. Miguel Morales, Grokking deep reinforcement learning. Manning Publications.
4. John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel, `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_, Last download 26/04/2022.
