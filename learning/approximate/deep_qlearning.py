import random

import numpy as np
from keras import Model
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from environment.environment import Environment
from learning.learningstrategy import LearningStrategy


class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # keras NN
    q2: Model  # keras NN

    def __init__(
            self,
            environment: Environment,
            batch_size: int,
            ddqn=False,
            λ=0.0005,
            γ=0.99,
            t_max=200,
    ) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn

        self.q1 = self.build_model(environment.state_size, environment.n_actions)
        self.q2 = self.build_model(environment.state_size, environment.n_actions)
        self.C = 10  # frequency of updating Q2 (how often the weights of the second neural network model (Q2) are updated with the weights from the first model (Q1))
        self.count = 0

    def build_model(self, state_size, n_actions):
        model = Sequential()
        model.add(Dense(64, input_dim=state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(n_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model

    def learn(self, episode):
        if episode.size >= self.batch_size:
            percepts = episode.sample(self.batch_size)
            self.learn_from_batch(percepts)
        super().learn(episode)

    def learn_from_batch(self, batch):
        D = self.build_training_set(batch)
        self.train_network(D)
        self.count += 1
        if self.count % self.C == 0:  # update Q2
            self.q2.set_weights(self.q1.get_weights())

    def build_training_set(self, percepts):
        D = []
        for percept in percepts:
            state = percept.state
            action = percept.action
            reward = percept.reward
            next_state = percept.next_state
            done = percept.done

            q_values = self.q1.predict(state.reshape(1, -1))
            q_max = np.max(self.q2.predict(next_state.reshape(1, -1)))
            q_target = reward if done else reward + self.γ * q_max
            q_values[0][action] = q_target
            D.append((state, q_values))
        return D

    #  FIX: TypeError: Can't instantiate abstract class DeepQLearning with abstract methods next_action, on_learning_startdef next_action(self, state):
    def train_network(self, D):
        for state, q_target in D:
            state = np.array(state).reshape(1, -1)  # reshape to 2D array
            self.q1.fit(state, q_target, epochs=1, verbose=0)
        state = np.reshape(D[-1][0], (1, -1))
        q_values = self.q1.predict(state)
        action = np.argmax(q_values)
        return action

    def on_learning_start(self):
        self.t = 0

    def next_action(self, state):
        if np.random.rand() <= self.ε:
            return random.randrange(self.env.n_actions)
        else:
            state = np.reshape(state, (1, -1))
            q_values = self.q1.predict(state)
            return np.argmax(q_values[0])
