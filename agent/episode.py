import random
from collections import deque

from agent.percept import Percept
from environment.environment import Environment


class Episode:
    """
    A collection of Percepts forms an Episode. A Percept is added per step/time t.
    The Percept contains the state, action, reward and next_state.
    This class is INCOMPLETE
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Return n final percepts from this Episode """
        return list(self._percepts)[-n:]

    def compute_returns(self) -> None:
        """ For EACH Percept in the Episode, calculate its discounted return Gt"""
        γ = 0.99
        gt = 0
        for percept in reversed(self._percepts):
            gt = percept.reward + γ * gt
            percept.return_value = gt

    def sample(self, batch_size: int):
        """ Sample and return a random batch of Percepts from this Episode """
        if len(self._percepts) < batch_size:
            return list(self._percepts)
        return random.sample(list(self._percepts), batch_size)

    @property
    def size(self):
        return len(self._percepts)
