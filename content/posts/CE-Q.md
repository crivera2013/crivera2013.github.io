---
title: "Correlated Q-Learning in Multi-Agent Games"
date: "2020-07-26"
template: "post"
draft: false
slug: "ceq-in-rl"
category: "Reinforcement Learning"
tags:
  - "Georgia Tech"
description: "Linear Programming + Game Theory = Better Multi-Agent Reinforcement Learning"
---
## Abstract
The purpose of this report is to re-implement one of the experiments found in the research paper, “Correlated Q-Learning”  (Greenwald & Hall, 2003).  The experiment in question is to implement four multi-agent Q-learning algorithms and compare their learning speeds in a zero-sum game referred to as “Soccer”.

## Introduction and Background
Multi-agent Q-learning is an area within Reinforcement Learning to design algorithms that can learn optimal policies in an environment containing other competitors.  While Markov Decision Processes (MDP) can be used to describe single-player, repeated games, a more generalized framework, a Markov Game, is necessary to describe a multi-agent game (Littman, 1994). These types of environments require a learning agent that can accounts for the possible actions of its opponents when determining the optimal move.  As such, Q-Learning in a Markov Game is described as defining a value given by a state paired with an action vector:

$$ Q_i(s, \vec a) = (1 - \alpha)R_i(s, \vec a) + \alpha \sum_{s'}P[s'|s,\vec a]V_i(s') $$

While this Markov Game framework is a strong starting point, in that it recognizes the impact of other agents’ decisions on the Q-Learning agent, more work needs to be done in defining a proper value function $V_i(s')$  that takes into account the true goals of the other agents.  A proper value function enables the Q-learning agent to correctly model the environment and other agents, have its Q-values converge, and therefore find an optimal policy.  Defining a proper value function is a difficult proposition with many proposed value functions only working in specific game settings.  For example, in two-player adversarial, constant-sum games, a Q-learning agent can apply a minimax algorithm to its value function in order to find an equilibrium policy.  This type of Q-Learning is known as Foe-Q (or alternatively Minimax-Q) (Littman, Friend-or-Foe Q-learning in General-Sum Games, 2001).  In contrast, Foe-Q will not work in games with collaborating agents and a better strategy would be to implement a maximax algorithm also known as Friend-Q (Littman, Friend-or-Foe Q-learning in General-Sum Games, 2001).  Both Friend-Q and Foe-Q only work in very specific circumstances and are, in effect, giving the Q-learning agent unearned information about an environment and its actors.  Greenwald, and Hall looked to solve this problem with a more generalized Q-learning approach which they referred to as Correlated Equilibrium Q-Learning (CE-Q).

$$ V_i(s) = \text{Nash}_i(s,Q_1,Q_2,...,Q_n) $$

What is interesting about Foe-Q and its use of the minimax algorithm is that you can frame the Q-Learning space as a payoff grid and treat the environment like a game theory problem.  This, in turn, enables the agent to use a Nash equilibrium within this payoff grid as the output of the value function.  Even better, the Nash equilibria can be discovered by treating the payoff grid as the constraint parameters for a linear programming optimization problem.  A Nash equilibrium is a vector of independent probability distributions that the Q-Learning agent can use to pick an optimal action.  This works if it assumes that opponent agents are only trying to minimize the Q-learning agents reward which works only in zero sum games.  In non-zero-sum games, the Nash equilibria fail in describing other agents’ actions where the other agents have multiple objectives: maximize their own scores and minimize the Q-learning agent’s score.  The authors of “Correlated Q-Learning” discovered a way to implement the concept of considering multi-objective opponents by considering the joint probabilities of each state for the opponents and Q-learning agent.  These joint probability constraints can be added to the original payoff grid constraints and this new correlated equilibrium can still be solved as a linear programming problem.  CE-Q thus enables a far more nuanced understanding by a Q-learning agent of other agents’ preferred actions beyond just the extremes of Foe-Q and Friend-Q and opens other potential value functions. Because of this, CE-Q acts as a generalization of both Foe-Q and Friend-Q while also enabling middle ground objective functions.  Greenwald, Hall, and Serrano propose 4 such functions and they experiment on one of them which they referred to as Utilitarian Correlated Equilibrium Q Learning (uCE-Q).

## Describing the Game and Q-Learning Experiments


In the paper, “Correlated Q-Learning”, the authors proceed to show CE-Q performance compared to regular Q-Learning, Foe-Q, and Friend-Q in several games including one referred to as “Soccer”.  This game consists of replicating a “1v1” faceoff scenario in a soccer game.  “Soccer” consists of a 2x4 grid space occupied by two players, A and B.   At any given point, one of the players has possession of the soccer ball and attempts to “score” by entering into the opposing player’s “goal” column (columns zero and three for A and B respectively).  This provides a reward of +100 however if the player enters its own goal while in possession of the ball, then the player’s reward is -100.  Both scenarios result in the end of the game, with any other state on the board providing a reward of zero.  Thus, “Soccer” is a zero-sum game.  Each player at any point can perform 5 actions (go north, south, east or west, or stay put).  Each player submits their action at the same time and the game environment decides at random who moves first thus making the environment stochastic.  If both players attempt to enter the same state and the current possessor of the “soccer ball” goes second, the possession changes to the other player.

#### *Soccer Game Environment*
![alt text](/media/soccer.png)

To compare the learning efficacy of each Q-Learning algorithm, the authors kept track of the error rate: the absolute difference between the Q value of a given state when the opponent stays put and the previous Q value of that state.  Each algorithm engaged in the game against a random action opponent until it has visited the given state 100,000 times.  The collected error rate values were then charted to observe whether the learning agent’s Q-values converged or not and how quickly they converged.

$$ \text{ERR}^t_i = |Q_i^t(s,\vec a) - Q_i^{t-1}(s, \vec a)| $$

## Assumptions Taken in Replicating the Experiments
Unfortunately, the paper, “Correlated Q-Learning”, left out some information regarding how to simulate the game, some edge case scenario rules of the game, how to measure the error rate, and hyperparameters of the game.  Because of this, I made the following assumptions:
- The opponent was not specified, so I decided that the opponent, player A, is a uniform random action opponent
- If the player who moved first decided to stay put and the second player attempted to move into the space, neither player would end up moving although, the player who attempted to move would now have possession of the ball
- If a player takes possession of the ball but while in his own goal column, that is still considered an own goal and the game is over.
- Every game starts with the same scenario with the agent as player B in position [0,1] with possession of the ball and player A in position [0,2].
- The paper does not specify exactly which state is used to log the Q-value error function but it infers that the state used is the starting scenario state so that is the state I used.
- I decided that the previous Q-value state was the value from the previous time step and not the time when that state was previously visited.
- While the paper specifies some hyperparameters (γ=0.9, α →0.001 for all algorithms and ε → 0.001 for the basic Q-Learning algorithm, it does not specify the starting α or decay factor which I specified as 0.5 and 0.9999, respectively.

## Experiment 1: Q-Learning Agent
![alt text](/media/q-learner.png)
The first experiment in the “Soccer” environment provides a baseline by seeing how a basic Q-Learning agent performs in a multi-agent environment.  Basic Q-learning does not take into consideration the actions of the other agents such that the state space is just every combination of the 112 states and 5 actions of the agent.  Since the Q-Learning agent is oblivious to its opponents, each Q-value is unstable since it does not consider how the opponent’s actions impacted the value of the state.  As such the Q-values are constantly being updated and never converge because they are trying to simultaneously value 25 different outcomes as just 5 outcomes for any given state.  We can see this failure to converge both in the original paper as well as in the replicated results.

## Experiment 2: Friend-Q Agent
![alt text](/media/friend-q.png)

$$ V_i(s) = \max_{\vec a \in A(s)} Q_i(s, \vec a) $$

The Friend-Q agent assumes that all other agents in the game are working in tandem with the Friend-Q agent to maximize its own rewards.  This is a naive and optimistic approach where it assumes that both agent and opponent
will either stay put or move east so that the player can score or benefit from the opponent’s own goal.  As such, the Q value for that state converges quite quickly which showed up in both the origins and my own experiments.  Since this is a zero-sum game, Friend-Q ends being nothing more than an expanded basic Q-Learning that considers the extra dimension created by the opposing player.

## Experiment 3: Foe-Q Agent

![alt text](/media/foe-q.png)

$$ V_i(s) = \max_{\sigma_1 \in \Sigma_1(s)} \min_{a_2 \in A_2(s)} Q_i(s, \sigma_1, a_2)$$

The Foe-Q algorithm has the Q-learning agent assume that the other agents will try to minimize its rewards which fits the description of this zero-sum game.  We see that the Foe-Q algorithm converges around 60,000 simulations in both the original and replicated experiment which suggests that Foe-Q is a viable Q-Learning method for zero sum games.  The Q-values do not converge as quickly for Foe-Q as this agent must grapple with the stochastic nature of its opponent and the randomness of who initiates a move first.

## Experiment 4: Utilitarian Correlated Equilibrium Agent
![alt text](/media/correlated-q.png)

$$ \sigma \in \text{argmax}_{\sigma ]in CE} \sum_{i \in I} \sum_{\vec a \in A} \sigma(\vec a)Q_i(s,\vec a)$$

As mentioned earlier, Correlated Equilibrium Q-learning provides more nuanced objective functions that enables the agent to consider opponents with multiple objectives.  Utilitarian CE-Q is one of those functions where the action the agent decides is summing up all the probability adjusted potential rewards for each action and choosing the action with the highest value.  As the results from both sets of experiments show, the uCE-Q agent also converges around 60,000 simulations exactly like the Foe-Q agent and it converges to the same optimal policy as the Foe-Q agent.  This is not a coincidence as in a zero-sum game, the objective function of the uCE-Q is the same as the minimax objective function of the Foe-Q agent.  This demonstration shows that CE-Q is a viable alternate as well as a strong generalization of Foe-Q and Friend-Q.

## Conclusion
Through these replicated Q-learning experiments, we observed multiple phenomena in Reinforcement learning.  We saw how a Markov Decision Process is not suitable for learning multi-agent games as the basic Q-Learning agent could not converge to a single policy.  This prompted an understanding of a more generalized Reinforcement Learning framework, Markov Games, that can handle multi-agent environments.  We observed Friend-Q and Foe-Q Markov Game implementations that converge in such environments before showing how Markov Game Q Learning can be generalized even further through the framework of Correlated Equilibrium Q learning.   CE-Q both solved the zero-sum “Soccer” game in the same manner as the Foe-Q algorithm while maintaining an ability to be applied to other constant-sum games and collaborative games where Foe-Q would fail.   Thus, CE-Q, as a generalization of Markov Games, opens new possibilities for Reinforcement Learning agents to better learn in unknown environments.

### Bibliography
- Greenwald, A., & Hall, K. (2003). Correlated Q-Learning. ICML.
- Littman, M. L. (1994). Markov Games as a Framework for Multi-Agent Reinforcement Learning. ICML.
- Littman, M. L. (2001). Friend-or-Foe Q-learning in General-Sum Games. ICML.









