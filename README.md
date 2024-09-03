# Frozen Lake and Cartpole

This repository has the solutions to the Frozen Lake and Cartpole reinforcement learning projects.

#### Frozen Lake

Frozen Lake is a simple environment composed of tiles, where the AI has to move from an initial tile to a goal. Tiles can be a safe frozen lake ✅, or a hole ❌ that gets you stuck forever. The AI, or agent, has 4 possible actions: go ◀️LEFT, 🔽DOWN, ▶️RIGHT, or 🔼UP.

#### Cartpole

The cartpole problem is an inverted pendulum problem where a stick is balanced upright on a cart. The cart can be moved left or right and the goal is to keep the stick from falling over. A positive reward of +1 is received for every time step that the stick is upright.

## About the project

The purpose of this project is to implement reinforcement learning using Q-learning, Deep Q-learning, N-step Q-learning, and Monte Carlo learning algorithms.

### Q-Learning

Q-learning is a machine learning algorithm that allows the model to learn and improve by trial and error. Good actions are rewarded or reinforced, while bad actions are discouraged and penalized.

Q-learning takes an off-policy approach to reinforcement learning. A Q-learning approach aims to determine the optimal action based on its current state. The Q-learning approach can accomplish this by either developing its own set of rules or deviating from the prescribed policy. A defined policy is unnecessary because Q-learning may deviate from the given policy.

The off-policy approach in Q-learning is achieved using Q-values -- also known as action values. The Q-values are the expected future values for action and are stored in the Q-table.

#### How does Q-learning work?
Q-learning models operate in an iterative process that involves multiple components working together to help train a model. The iterative process involves the agent learning by exploring the environment and updating the model as the exploration continues. The multiple components of Q-learning include the following:

- Agents. The agent is the entity that acts and operates within an environment.
- States. The state is a variable that identifies the current position in the environment of an agent.
- Actions. The action is the agent's operation when it is in a specific state.
- Rewards. A foundational concept within reinforcement learning is the concept of providing either a positive or a negative response to the agent's actions.
- Episodes. An episode is when an agent can no longer take a new action and ends up terminating.
- Q-values. The Q-value is the metric used to measure an action at a particular state.

Q-learning models work through trial-and-error experiences to learn the optimal behavior for a task. The Q-learning process involves modeling optimal behavior by learning an optimal action-value function or q-function. This function represents the optimal long-term value of action a in state s and subsequently follows optimal behavior in every subsequent state.

Bellman's equation
```
Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
```

The equation breaks down as follows:

- Q(s, a) represents the expected reward for taking action a in state s.
- The actual reward received for that action is referenced by r while s' refers to the next state.
- The learning rate is α and γ is the discount factor.
- The highest expected reward for all possible actions a' in state s' is represented by max(Q(s', a')).
  
#### What is a Q-table?
The Q-table includes columns and rows with lists of rewards for the best actions of each state in a specific environment. A Q-table helps an agent understand what actions are likely to lead to positive outcomes in different situations.

The table rows represent different situations the agent might encounter, and the columns represent the actions it can take. As the agent interacts with the environment and receives feedback in the form of rewards or penalties, the values in the Q-table are updated to reflect what the model has learned.

The purpose of reinforcement learning is to gradually improve performance through the Q-table to help choose actions. With more feedback, the Q-table becomes more accurate so the agent can make better decisions and achieve optimal results.

The Q-table is directly related to the concept of the Q-function. The Q-function is a mathematical equation that looks at the current state of the environment and the action under consideration as inputs. The Q-function then generates outputs along with expected future rewards for that action in the specific state. The Q-table allows the agent to look up the expected future reward for any given state-action pair to move toward an optimized state.

#### What is the Q-learning algorithm process?
The Q-learning algorithm process is an interactive method where the agent learns by exploring the environment and updating the Q-table based on the rewards received.

The steps involved in the Q-learning algorithm process include the following:
- Q-table initialization. The first step is to create the Q-table as a place to track each action in each state and the associated progress.
- Observation. The agent needs to observe the current state of the environment.
- Action. The agent chooses to act in the environment. Upon completion of the action, the model observes if the action benefits the environment.
- Update. After the action has been taken, it's time to update the Q-table with the results.
- Repeat. Repeat steps 2-4 until the model reaches a termination state for a desired objective.

### Deep Q-Learning

Q-Learning is required as a pre-requisite as it is a process of Q-Learning creates an exact matrix for the working agent which it can “refer to” to maximize its reward in the long run. Although this approach is not wrong in itself, this is only practical for very small environments and quickly loses it’s feasibility when the number of states and actions in the environment increases. The solution for the above problem comes from the realization that the values in the matrix only have relative importance ie the values only have importance concerning the other values. Thus, this thinking leads us to Deep Q-Learning which uses a deep neural network to approximate the values. This approximation of values does not hurt as long as the relative importance is preserved. The basic working step for Deep Q-Learning is that the initial state is fed into the neural network and it returns the Q-value of all possible actions as an output. The difference between Q-Learning and Deep Q-Learning can be illustrated as follows:

Deep Q-Learning is a type of reinforcement learning algorithm that uses a deep neural network to approximate the Q-function, which is used to determine the optimal action to take in a given state. The Q-function represents the expected cumulative reward of taking a certain action in a certain state and following a certain policy. In Q-Learning, the Q-function is updated iteratively as the agent interacts with the environment. Deep Q-Learning is used in various applications such as game playing, robotics, and autonomous vehicles.

Deep Q-Learning is a variant of Q-Learning that uses a deep neural network to represent the Q-function, rather than a simple table of values. This allows the algorithm to handle environments with a large number of states and actions, as well as to learn from high-dimensional inputs such as images or sensor data.

One of the key challenges in implementing Deep Q-Learning is that the Q-function is typically non-linear and can have many local minima. This can make it difficult for the neural network to converge to the correct Q-function. To address this, several techniques have been proposed, such as experience replay and target networks.

Experience replay is a technique where the agent stores a subset of its experiences (state, action, reward, next state) in a memory buffer and samples from this buffer to update the Q-function. This helps to decorrelate the data and make the learning process more stable. Target networks, on the other hand, are used to stabilize the Q-function updates. In this technique, a separate network is used to compute the target Q-values, which are then used to update the Q-function network.

### N-Step Q-Learning

The N-step Q learning algorithm works in a similar manner to DQN except for the following changes:

No replay buffer is used. Instead of sampling random batches of transitions, the network is trained every N steps using the latest N steps played by the agent.

In order to stabilize the learning, multiple workers work together to update the network. This creates the same effect as uncorrelating the samples used for training.

Instead of using single-step Q targets for the network, the rewards from $N$ consequent steps are accumulated to form the N
-step Q targets, according to the following equation: 
```
R(st,at)=∑i=t+k−1i=tγi−tri+γkV(st+k)
```
Parameters:
- num_steps_between_copying_online_weights_to_target – (StepMethod) The number of steps between copying the online network weights to the target network weights.
- apply_gradients_every_x_episodes – (int) The number of episodes between applying the accumulated gradients to the network. After every num_steps_between_gradient_updates steps, the agent will calculate the gradients for the collected data, it will then accumulate it in internal accumulators, and will only apply them to the network once in every apply_gradients_every_x_episodes episodes.
- num_steps_between_gradient_updates – (int) The number of steps between calculating gradients for the collected data. In the A3C paper, this parameter is called t_max. Since this algorithm is on-policy, only the steps collected between each two gradient calculations are used in the batch.
- targets_horizon – (str) Should be either ‘N-Step’ or ‘1-Step’, and defines the length for which to bootstrap the network values over. Essentially, 1-Step follows the regular 1 step bootstrapping Q learning update.

### Monte Carlo Algorithm 

Any method that solves a problem by generating suitable random numbers, and observing that fraction of numbers obeying some property or properties, can be classified as a Monte Carlo prediction reinforcement learning method.

The method works by running simulations or episodes where an agent interacts with the environment until it reaches a terminal state. At the end of each episode, the algorithm looks back at the states visited and the rewards received to calculate what’s known as the “return” — the cumulative reward starting from a specific state until the end of the episode. Monte Carlo policy evaluation repeatedly simulates episodes, tracking the total rewards that follow each state and then calculating the average. These averages give an estimate of the state value under the policy being followed.

By aggregating the results over many episodes, the method converges to the true value of each state when following the policy. These values are useful because they help us understand which states are more valuable and thus guide the agent toward better decision-making in the future. Over time, as the agent learns the value of different states, it can refine its policy, favoring actions that lead to higher rewards.

#### Mathematical Concepts in Monte Carlo Policy Evaluation:
In Monte Carlo policy evaluation, the value V of a state “s” under a policy π is estimated by the average return G following that state. The return is the cumulative reward obtained after visiting state “s”:
```
V(s) = \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i     
```

Here, N(s) is the number of times state “s” is visited across episodes, and Gi is the return from the i-th episode after visiting state “s”. This average converges to the expected return as N(s) becomes large:
```
V(s) \approx E_{\pi}[G|S=s]  
```

Each return Gi is calculated by summing discounted rewards from the time state “s” is visited till the end of the episode:
```
G_i = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
```
γ is the discount factor (between 0 and 1) and R is the reward at each time step. This reflects the idea that rewards in the near future are more valuable than rewards further in the future.

## Features

OpenAI Gym: a Pythonic API that provides simulated training environments to train and test reinforcement learning agents

## Installation

1. Clone the repository
   ```
   git clone https://github.com/alecsiuh/reinforcement-learning.git
   cd reinforcement-learning
   ```
2. Create a virtual environment
   ```
   python3 -m venv venv
   venv\Scripts activate
   ```
3. Install the required dependencies
   ```
   pip install -r requirements.txt
   ```

## Technologies used
Python: used to implement the algorithms.

## License 
Code completed by Alexia Cismaru.

## Sources
TechTarget: https://www.techtarget.com/searchenterpriseai/definition/Q-learning

GeeksForGeeks: https://www.geeksforgeeks.org/deep-q-learning/, https://www.geeksforgeeks.org/monte-carlo-policy-evaluation/

GitHub: https://intellabs.github.io/coach/components/agents/value_optimization/n_step.html
