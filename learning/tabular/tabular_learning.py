from abc import abstractmethod

import numpy as np
from agent.episode import Episode
from environment.environment import Environment
from learning.learningstrategy import LearningStrategy
from numpy import ndarray, random


class TabularLearner(LearningStrategy):
    """
    A tabular learner implements a tabular method such as Q-Learning, N-step Q-Learning, ...
    """

    π: ndarray
    v_values: ndarray
    q_values: ndarray

    def __init__(
            self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99
    ) -> None:
        super().__init__(environment, λ, γ, t_max)
        # learning rate
        self.α = α
        # policy table
        self.π = np.full(
            (self.env.n_actions, self.env.state_size), fill_value=1 / self.env.n_actions
        )
        # state value table
        self.v_values = np.zeros((self.env.state_size,))
        # state-action table
        self.q_values = np.zeros((self.env.state_size, self.env.n_actions))

    @abstractmethod
    def learn(self, episode: Episode):
        # subclasses insert their implementation at this point
        # see for example be\kdg\rl\learning\tabular\qlearning.py
        self.evaluate()
        self.improve()
        super().learn(episode)

    def on_learning_start(self):
        self.t = 0

    def next_action(self, s: int):
        action_probabilities = self.π[:, s].tolist()
        actions = list(range(len(action_probabilities)))

        if np.random.random() < self.ε:
            chosen_action = np.random.choice(actions, p=action_probabilities)
        else:
            max_prob = max(action_probabilities)
            tied_actions = [a for a, prob in enumerate(action_probabilities) if prob == max_prob]
            # apply tie-breaking
            if len(tied_actions) > 1:
                chosen_action = random.choice(tied_actions)
            else:
                chosen_action = tied_actions[0]
        return chosen_action

    def evaluate(self):
        for s in range(len(self.v_values)):
            # q_values[s, :] is a 1D array that corresponds to a row in the q_values[s, a] table
            # q_values[s, :] represents the action that has the best quality
            self.v_values[s] = np.max(self.q_values[s, :])

    def improve(self):
        for s in range(self.env.state_size):
            best_a = np.argmax(self.q_values[s, :])
            for a in range(self.env.n_actions):
                if best_a == a:
                    # probability that the best action will be taken
                    self.π[a, s] = 1 - self.ε + self.ε / self.env.n_actions
                else:
                    # the probability that a random action will be taken
                    self.π[a, s] = self.ε / self.env.n_actions
        # tau increases only when the episode ends
        self.decay()
