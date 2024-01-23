from agent.agent import Agent, TabularAgent, DQL
from environment.openai import FrozenLakeEnvironment
from learning.tabular.qlearning import Qlearning, NStepQlearning, MonteCarloLearning
from environment.openai import CartPoleEnvironment
from learning.approximate.deep_qlearning import DeepQLearning

if __name__ == '__main__':
    frozen_lake = FrozenLakeEnvironment(animate=False)
    cart_pole = CartPoleEnvironment(animate=False)

    # create an Agent that uses Qlearning Strategy
    # agent: Agent = TabularAgent(frozen_lake, Qlearning(frozen_lake))
    # agent.train()

    # # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(frozen_lake, NStepQlearning(frozen_lake, 5))
    # agent.train()

    # # create an Agent that uses MonteCarloLearning Strategy
    # agent: Agent = TabularAgent(frozen_lake, MonteCarloLearning(frozen_lake, 5))
    # agent.train()

    # create an Agent that uses DeepQLearning Strategy
    agent: Agent = DQL(cart_pole, DeepQLearning(cart_pole, 20))
    agent.train()
