import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

frozen_lake_policy = "/frozen_lake/policy"
if not os.path.exists(frozen_lake_policy):
    os.makedirs(frozen_lake_policy)

frozen_lake_q = "/frozen_lake/q_values"
if not os.path.exists(frozen_lake_q):
    os.makedirs(frozen_lake_q)

cart_pole = "/cartpole"
if not os.path.exists(cart_pole):
    os.makedirs(cart_pole)


class ChartDemo:
    #### CHARTS FOR FROZEN LAKE ####
    def plot_q_values(self, q_values):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.subplot(1, 2, 1)
        plt.suptitle(f'Time: {current_time}')
        plt.title('q-values')
        sns.heatmap(q_values, cmap="Blues", annot=True, cbar=False, square=False, vmin=0, vmax=1)
        plt.savefig(f'frozen_lake/q_values/q_values_{current_time}.png')
        # avoid overlapping of figures
        plt.clf()

    def plot_policy(self, policy):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.subplot(1, 2, 2)
        plt.title(f'Policy π - Time: {current_time}')
        policy_by_action = np.argmax(np.transpose(policy), 1)
        action_mapping = {
            0: [-1, 0],  # Left
            1: [0, -1],  # Down
            2: [1, 0],  # Right
            3: [0, 1]  # Up
        }

        xy_values = [action_mapping[value] for value in policy_by_action]
        x_values = np.reshape([col[0] for col in xy_values], (4, 4))
        y_values = np.reshape([col[1] for col in xy_values], (4, 4))
        plt.quiver(x_values, y_values)
        plt.savefig(f'frozen_lake/policy/policy_{current_time}.png')
        plt.clf()
