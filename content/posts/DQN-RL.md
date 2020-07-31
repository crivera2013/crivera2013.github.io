---
title: "Deep Q-Network in Reinforcement Learning"
date: "2020-07-02"
template: "post"
draft: false
slug: "dqn-in-rl"
category: "Reinforcement Learning"
tags:
  - "Georgia Tech"
description: "Neural Networks combined with Reinforcement Learning for all of the math...."
---

## Abstract
The purpose of this paper is to experiment with a Deep Q-Network (DQN) reinforcement learning (RL) algorithm in order to find the optimal policy for a given environment.  For off-policy RL, agents use Q-Learning in order to create a policy that, for every state in an environment, a Q-learning algorithm will return the optimal action for that state.  While Q-Learning is a powerful RL method, it has significant limitations.  The first limitation is that Q-learning methods are viable only in environments with small, discrete state spaces.  In near-infinite and/or continuous state spaces, a Q-learning matrix would be prohibitively hard to map.  For example,  Q-learning would require a prohibitive amount of time and memory to explore the entire $10^{46}$ sized state space for a game of Chess.  We can overcome this hurdle by applying function approximations to generalize a Q-Learning space.  In this paper we will be applying artificial neural networks (ANN) as the Q-Learning function approximator otherwise known as a Deep Q-Network.

## Introduction and Background
This paper explores the implementation and hyperparameter tuning of a Deep Q-Network  algorithm in order to create an agent that can solve the game “Lunar Lander”.  In this game, an agent is presented with a virtual spaceship with the goal of landing the ship on the ground within a landing zone with the correct orientation (feet first).  Landing the ship correctly provides positive rewards while landing the ship away from the landing zone or in the wrong orientation creates negative rewards.  The agent is considered to have learned the optimal policy if it has successfully landed the spaceship in the landing zone the majority of the time in a given one hundred episodes.  This game includes a non-trivial state space where an RL agent receives information regarding six continuous values and two binary values.  For a state space like this, function approximators are needed which and where this paper uses ANNs to solve the problem.
![alt text](/media/lunarLander.png)

A function approximator replaces a Q-learning matrix with weights from a supervised learning model.  The supervised learning model used in this paper is a two-layer neural network constructed using the Python package PyTorch.  A neural network can be thought of as a collection of interconnected logistic regressions that enable models to identify non-linear relationships in the data.  This would generally cause issues of non-convergence for an RL agent however Google DeepMind discovered a combination of two methods to solve the problem: “experience replay” and a delayed “iterative update”.  With “experience replay”, an RL agent keeps a record of every $(S_t,A_t,R,S_{t+1})$ and will randomly sample batches of these records for retraining which minimizes the negative impact that action sequence correlations have on a neural network’s weights.  The “iterative update” method is the function approximation equivalent of temporal difference (TD) learning in classical Q-Learning where the ANN model computes the loss function by comparing the predictions of an earlier version of the model’s weights with the estimates of its current weights.

For the experiments in this paper,  I wanted to observe how changing the hyperparameters for “experience replay” and “iterative update” impacted the DQN agent’s ability to discover the optimal policy of the “Lunar Lander” game.  The hyperparameters were judged by both how many episodes the model needed until it averaged a score of 200 or above over the last 100 episodes.

## Experiment 1: Changing the batch size of the experience replay

Experience replay provides an RL agent several advantages: it enables the agent to reuse data it has already seen to increase efficiency and it removes undesirable weight changes caused by the correlation of consecutives steps in an episode.  This correlation prevents convergence or can isolate an agent at a local optimum.  For the experiment, I had the agent learn the environment with different experience replay batch sizes: [4, 8, 16, 32, 64, 128, 256, 512].

![alt text](/media/batchTrain1.png)

![alt text](/media/batchTrain2.png)

![alt text](/media/batchTest1.png)
As the graphs above show, the agents with the larger batch sizes (128, 256, or 512) were able to learn the environment in roughly 1650 episodes while the smaller batch sizes needed 1900 or more episodes to acquire equivalent learning.  These results support the findings from the Google Deepmind team that agents reusing states they already explored reduces the number of episodes needed and enables faster convergence.  These larger batch sizes also seemed to limit overcorrection of the weights as the average rolling mean scores of the larger batch-sized agents during training were significantly less volatile than their peer agents with lower batch sizes.

## Experiment 2: Changing the frequency of the experience replay

Another hyperparameter I experimented on for the DQN agent’s experience replay was how frequently the agent engaged with experience replay during training.  I wanted to compare the number of actions an agent takes before using experience replay with how many episodes it takes the agent to learn the environment.  For this experiment, I tested agents that activate experience replays for frequencies ranging from 1-12 actions.

![alt text](/media/replayTrain1.png)

![alt text](/media/replayTrain2.png)

![alt text](/media/replayTest1.png)

This experiment also reinforces the value of experience replay as there clear relationship in the data showing that, as the frequency of experience replays  increases, the number of episodes needed to train the agent decreases.  Inserting experience replay between the actions of an agent does appear to minimize  correlation impact of consecutive actions on an agent’s neural network weights.  At the same time, increasing replay step frequency, unlike increasing batch size, does not minimize learning volatility.  The faster learner had a replay step frequency of one but also had the highest learning volatility with sharp upticks in knowledge gains (episodes 800-900 above) but also bouts of flat or negative knowledge gains (episodes 1000-1250).
We can further observe the lack of impact that changing the frequency replay steps has on an agent’s volatility in an environment by observing the test results for the different replay step agents.  While the agent with a replay step frequency of 1 is the least volatile, it does not have appreciably better performance in the testing phase.  This suggests that, while having a tight frequency of replay steps is preferable, it is less important of a hyperparameter than experience replay batch size when tuning a DQN RL agent.

## Why experience replay makes sense in the context of the Lunar Lander
As the results so far have shown, with both high frequency replay steps and large batch sizes, experience replay-enhanced RL agents perform better in the Lunar Lander scenario.  As mentioned previously, experience replay prevents overfitting caused by the correlation effect of consecutive steps in.  In the Lunar Lander scenario, this overfitting manifests itself in poor results due to the fact that the “landing zone” is located in the middle of the screen and because the agent does not receive any information regarding acceleration.  The agent is only given information regarding velocity and position which leads to over-correction caused by greedy immediate knowledge.  For example, If the agent starship position was located on the far left corner of the screen, the agent would be rewarded for activating the left booster thus pushing it closer to center and landing zone.  The agent would repeatedly activate the left booster to get more reward but, at a certain point, would have so much acceleration that the otherwise small left booster activation would launch the agent to the far right corner, missing the center, and receiving a large negative reward.  An agent with experience replay would hold a diminished value for repeatedly activating the left booster and would then not overshoot the landing zone.

## Experiment 3: Changing the learning rate for iterative update

![alt text](/media/learningRateTrain.png)

![alt text](/media/learningRateTest.png)

A component of a neural network model is the optimization function it uses to update its weights during training.  For the DQN agent, I went with using an Adam optimizer and tuned the learning rate hyperparameter in order to investigate its impact on the agent’s training success.  A learning rate is a dampening effect on an ANN’s weight update such that the ANN does not over correct itself.  For this experiment, I tried multiple learning rates ranging from $1x10^{-5}$ to $1x10^{-2}$ and found that the RL agents that failed to solve the environment within 3000 episodes for learning rates both too high and too low.  For a learning rate of $1x10^{-5}$, the agent got stuck in a local optima with a score of -100 while a learning rate of $1x10^{-2}$ was too high and caused overcorrections on weight changes denoted by its much higher volatility than the other.
The best performing agent had a learning rate of $1x10^{-3}$ as it learned the fastest, had the lowest training volatility and the highest average test scores.  What is interesting here is that $1x10^{-3}$ is the default learning rate parameter for the Adam optimizer which supports the “adaptive moment estimation” concept of this optimizer.  The Adam optimizer can apply limited dynamic changes to its learning rate towards first and second moments in the weight changes.  The value $1x10^{-3}$ is a middle ground enabling the Adam optimizer to both increase and decrease the learning rate.  This optimizer is thus useful in sparse environments like the Lunar Lander scenario where the agent makes anywhere between 200-500 actions before receiving non-trivial positive or negative rewards.  The results from this experiment reinforce the power of the Adam optimizer as a strong choice for ANN and DQN reinforcement learning.

## Optimized Model Results

![alt text](/media/bestModelTrain.png)

![alt text](/media/bestModelTest.png)

For the optimized RL model we can observe how the model’s score improves over time as the number of completed episodes increases.  While the model does not always score above 200 (landing the spaceship in the landing zone), it clearly does so with increased frequency and does quite well in the testing session.  In the 100 testing episodes, the RL agent achieved a negative score only twice and had a score of 200 or more 71% of the time.  With even more training, the model’s performance would improve even more but, at this point, these results show the effectiveness of DQN agents.

## Conclusion
In this paper, we observed how artificial neural networks can be applied as Q-learning function approximators in order for reinforcement learning agents to solve non-trivial Partially-Observable Markov Decision Processes.  We experimented with tuning the hyperparameters of the experience replay and iterative update algorithms that make Deep Q-Learning possible and saw how they impacted an RL agent’s ability to learn an environment.