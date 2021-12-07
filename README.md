# RL anonymity

An experimental effort to use reinforcement learning techniques for data anonymity. 

## Conceptual overview

Reinforcement learning is a learning framework based on accumulated experience. In this paradigm, an agent is learning by iteracting with an environment 
without (to a large extent) any supervision. The following image   schematically describes the reinforcement learning framework 

![RL paradigm](images/agent_environment_interface.png "Reinforcement learning paradigm") 

The framework has been use successfully to many recent advances in corntol, robotics, games and elsewhere.

Given that data anonymity is essentially an optimization problem; between data utility and privacy, in this repository we try
to use the reinforcement learning paradigm in order to train agents to perform this optimization for us. The following image
places this into a persepctive 


![RL anonymity paradigm](images/general_concept.png "Reinforcement learning anonymity schematics") 

