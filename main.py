import numpy as np
from matplotlib import pyplot as plt

from QLearning import QLearning
from BalanceTask import BalanceTask
from BalanceMoveTask import BalanceMoveTask

task = BalanceTask()
# task = BalanceMoveTask(0,5)
env = task.env

qlearning = QLearning(env,task, 9)

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
for irehearsal in range(7000):

    # Learn.
    # ------
    r = qlearning.do_episode()

    if irehearsal % 50 == 0:
        # Perform (no learning).
        # ----------------------
        # Perform.
        qlearning.do_episode_withoutLearning()
        perform_cumreward = qlearning.last_rewardcumul
        perform_cumrewards.append(perform_cumreward)


        ax1.cla()
        ax1.plot(perform_cumrewards, '.--')
        # Wheel trajectories.
        update_wheel_trajectories()

        plt.pause(0.001)