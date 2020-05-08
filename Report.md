

## Overview

The goal of this project is to achieve an average reward of at least +30 from all agents (20) over 100
consecutive episodes in the environment.
The environment is a double-jointed arm that can move to a target.
A reward of +0.1 is given at every step when the agent hand is in the target location. The goal of
the agent is to maintain its position on the target for as long as possible.
The state space is a vector of 33 variables corresponding to the position, rotation, velocity and
angular velocities of the arm.
The action space is a vector with four numbers between -1 and 1 corresponding to torque
applied to two joints. 

## Implementation
The project was solved using Deep Reinforcement Learning, more specifically using Deep
Deterministic Policy Gradient or DDPG. The code was based upon the ddpg-bipedal and ddpgpendulum example from the Udacity Deep Reinforcement Learning GitHub repo. This was
modified and updated to work with the Unity-ML environment and extended with new model
architecture. Some of the code was separated out to modules, for example ReplayBuffer class
in memory.py.
The Jupyter notebook results.ipynb contains the implementation for training the agent in the
environment.
Agent.py contains a DDPG agent which interacts with the environment to optimize the reward.
Memory.py contains the Replay Buffer which is used by the Agent to record and sample a
(state, action, reward, next_state) tuples for training the model.
Model.py contains the Actor and Critic Neural Network Class which takes in the input state and
outputs the desired Q-values
Noise.py contains the OUNoise Class which contains the Ornstein-Uhlenback process which
adds noise to actions. 


## Learning algorithm
I used the DDPG learning algorithm based on the [Continous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) paper (Lillicrap et al., 2016). The starting code was based on the Bipendulum implementation introduced during the class.

This is because it is different to normal Actor Critic algorithms in that the
Critic is used to approximate the maximizer over the Q-Values of the next state and not a learnt
baseline. It uses a Replay Buffer and is Off-Policy and Model-Free. At its core, DDPG is a policy
gradient algorithm that uses a stochastic behavior policy for good exploration but estimates
a deterministic target policy, which is much easier to learn 

DDPG is an actor-critic method where the Critic learns from the value function and it determines how the Actor policy improves. To decrease the instability of the model, I used a replay buffer and a soft target update. 

## Model architecture
The actor model is a neural network with three hidden layer with size 256, 128 and 64. I used ReLU as the activation function and tanh is used in the final layer to return the action output.

The critic model is a neural network with four hidden layers of size 256, 256, 128 and 64. I used ReLU as the activation function and tanh is used in the final layer to return the action output.

## DDPG
The DDPG function in results.ipynb brings all these components together to train the network.
It is basically a runner for the DDPG algorithm in the Agent class. The agent interacts with the
environment by viewing the state and performing an action based on the policy from the DDPG.
The rewards and next state are then sent back to the agent. The agent records a tuple (s,a,r,s’)
into replay memory. It then samples these to updates the Actor and Critic networks to optimize
rewards. This process is done until the desired average score is achieved over all agents which is
30+, it then discontinues training and finishes the loop. 

## Training plots
I solved the environment in 150 episodes. I used Udacity's GPU and it took me around 8-10 hours to solve the environment.

https://github.com/EngrZainabAkhtar/DRL-Continuous-Control/edit/master/Report.md/graph_showing_30.12.png

Here it shows that 30.12 rate is achieved.

## Training
The training done on 150 episodes and plotted as shown in the html file.

## Ideas for future work

The project has given me a good understanding of the difficulties of a continuous control
problem in Reinforcement Learning.
Training the single agent version of this environment was simple but the 20-agent brought new
challenges.
Changing the update rate to eg 2 or 4 increased stability but slowed down training. Finding a
happy medium with this was hard so I set it to 1
In the end, I modified the noise and found a great model and hyperparameters that trained
quickly, steady overall learning curve and average over all agents well over each episode. should improve the learning speed of the algorithm. In addition to that, I could use the simulator with 20 agents to speed up learning.
