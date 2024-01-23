import numpy as np

from agent.episode import Episode
from charts.visuals import ChartDemo
from environment.environment import Environment
from learning.tabular.tabular_learning import TabularLearner


class Qlearning(TabularLearner):
    def __init__(
            self, environment: Environment, α=0.7, λ=0.0005, γ=0.9, t_max=99
    ) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)

    def visualize_learning(self):
        chart = ChartDemo()
        chart.plot_q_values(self.q_values)
        chart.plot_policy(self.π)

    def learn(self, episode: Episode):
        percept = episode.percepts(0)[-1]
        s = percept.state
        a = percept.action
        r = percept.reward
        next_s = percept.next_state

        self.q_values[s, a] = self.q_values[s, a] + self.α * (
                r + self.γ * np.max(self.q_values[next_s, :]) - self.q_values[s, a])

        if episode.size % 1 == 0:  # plot at the end of every episode
            self.visualize_learning()
        super().learn(episode)


class NStepQlearning(TabularLearner):
    def __init__(
            self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99
    ) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def visualize_learning(self):
        chart = ChartDemo()
        chart.plot_q_values(self.q_values)
        chart.plot_policy(self.π)

    def learn(self, episode):
        # check if the episode contains enough n steps
        if episode.size >= self.n:
            # check for the percepts in reverse because we need to start from the last percept
            for p in reversed(episode.percepts(self.n)):
                s = p.state
                a = p.action
                r = p.reward
                next_s = p.next_state
                done = p.done
                self.q_values[s, a] = self.q_values[s, a] - self.α * (
                        self.q_values[s, a] - (r + self.γ * np.max(self.q_values[next_s, :]))
                )

        self.visualize_learning()
        super().learn(episode)


class MonteCarloLearning(TabularLearner):
    def __init__(
            self, environment: Environment, n: int, α=0.7, λ=0.0005, γ=0.9, t_max=99
    ) -> None:
        TabularLearner.__init__(self, environment, α, λ, γ, t_max)
        self.n = n  # maximum number of percepts before learning kicks in
        self.percepts = []  # this will buffer the percepts

    def visualize_learning(self):
        chart = ChartDemo()
        chart.plot_q_values(self.q_values)
        chart.plot_policy(self.π)

    def learn(self, episode: Episode):
        # percepts = episode.percepts()
        # for percept in percepts:
        #     s = percept.state
        #     a = percept.action
        #     Gt = percept.return_
        #
        #     print(a)
        #     print(Gt)
        #
        #     self.q_values[s, a] += self.α * (Gt - self.q_values[s, a])
        #     self.π[s, :] = 0
        #     self.π[s, np.argmax(self.q_values[s, :])] = 1
        # super().learn(episode)
        # print(self.q_values)

        for p in reversed(episode.percepts(self.n)):
            s = p.state
            a = p.action
            r = p.reward
            G = r  # Initialize G with the immediate reward
            done = p.done
            self.q_values[s, a] = self.q_values[s, a] + self.α * (G - self.q_values[s, a])

            if done:
                self.visualize_learning()
        super().learn(episode)
