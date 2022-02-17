Conceptual overview
===================

The term data anonymization refers to techiniques that can be applied on a given dataset, D, such that after
the latter has been submitted to such techniques, it makes it difficult for a third party to identify or infer the existence
of specific individuals in D. Anonymization techniques, typically result into some sort of distortion
of the original dataset. This means that in order to maintain some utility of the transformed dataset, the transofrmations
applied should be constrained in some sense. In the end, it can be argued, that data anonymization is an optimization problem
meaning striking the right balance between data utility and privacy. 

Reinforcement learning is a learning framework based on accumulated experience. In this paradigm, an agent is learning by iteracting with an environment 
without (to a large extent) any supervision. The following image describes, schematically, the reinforcement learning framework .

![RL paradigm](images/agent_environment_interface.png "Reinforcement learning paradigm") 

The agent chooses an action, ```a_t```, to perform out of predefined set of actions ```A```. The chosen action is executed by the environment
instance and returns to the agent a reward signal, ```r_t```, as well as the new state, ```s_t```, that the enviroment is in. 
The framework has successfully been used  to many recent advances in control, robotics, games and elsewhere.


Let's assume that we have in our disposal two numbers a minimum distortion, ```MIN_DIST``` that should be applied to the dataset
for achieving privacy and a maximum distortion, ```MAX_DIST```,  that should be applied to the dataset in order to maintain some utility.
Let's assume also that any overall dataset distortion in ```[MIN_DIST, MAX_DIST]``` is acceptable in order to cast the dataset as 
preserving  privacy and preserving dataset utility. We can then train a reinforcement learning agent to distort the dataset
such that the aforementioned objective is achieved.
