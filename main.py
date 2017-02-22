import numpy as np
from matplotlib import pyplot as plt

from QLearning import QLearning
from Sarsa_lambda import Sarsa_lambda
from BalanceTask import BalanceTask, LinearFATileCodingBalanceTask
from BalanceMoveTask import BalanceMoveTask, LinearFATileCodingBalanceMoveTask
import seaborn

task = LinearFATileCodingBalanceTask()
# task = LinearFATileCodingBalanceMoveTask(10, 10)
env = task.env

# learning = QLearning(env, task, 9, K_discountFactor=2000, epsilon_min=0.3, gamma=0.99, epsilon_decay=0.985, Sarsa=True)
learning = Sarsa_lambda(env, task, 9, alpha=0.5, epsilon_min=0.3,epsilon=0.1, gamma=0.8, epsilon_decay=1, lambd=0.9)
env.saveWheelContactTrajectories(True)
plt.ion()
plt.figure(figsize=(8, 4))

ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)


def update_wheel_trajectories():
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')


perform_cumrewards = []
for irehearsal in range(2000000):

    # Learn.
    # ------
    r = learning.do_episode()

    if irehearsal % 50 == 0:
        # Perform (no learning).
        # ----------------------
        # Perform.
        learning.do_episode_withoutLearning()
        perform_cumreward = learning.last_rewardcumul
        perform_cumrewards.append(perform_cumreward)


        ax1.cla()
        ax1.plot(perform_cumrewards, '.--')
        # Wheel trajectories.
        update_wheel_trajectories()

        plt.pause(0.001)