import numpy as np
from matplotlib import pyplot as plt

from QLearning import QLearning
from Sarsa_lambda import Sarsa_lambda
from BalanceTask import BalanceTask, LinearFATileCodingBalanceTask
import itertools
from BalanceMoveTask import BalanceMoveTask, LinearFATileCodingBalanceMoveTask
import seaborn

"""
 In this main, we only learn to balance the bicycle. You can change the parameter if you want
"""

task = LinearFATileCodingBalanceTask()
env = task.env

# learning = QLearning(env, task, 9, alpha=0.5, epsilon_min=0.0,epsilon=1, gamma=0.9, epsilon_decay=0.999, lambd=0.7,Sarsa=True, metrique_reward=0)
learning = Sarsa_lambda(env, task, 9, alpha=0.05, epsilon_min=0.0,epsilon=1, gamma=0.9, epsilon_decay=0.999, lambd=0.7, metrique_reward=0)
env.saveWheelContactTrajectories(True)
plt.ion()




fig1 = plt.figure()
ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()
fig3 = plt.figure()
ax3 = fig3.gca()

def update_wheel_trajectories():
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')

params = ["alpha : ", learning.alpha, "K_disconout_factor : ", learning.K_discountFactor, "epsilon_min : ", learning.epsilon_min, "gamma : ", learning.gamma,\
                                       " epsilon_decay : ", learning.epsilon_decay, "lambda : ", learning.lambd, " metrique : ", learning.metrique_reward]
print (params)
perform_cumrewards = []
number_itterations = []
episodes = []
number_episodes = 7000
for irehearsal in itertools.count():
    if learning.num_episode > number_episodes:
        break

    # Learn.
    # ------
    r = learning.do_episode()

    if irehearsal % 10 == 0:
        # Perform (no learning).
        # ----------------------
        # Perform.
        last_cumulrewards = []
        last_t = []
        for t in range(3):
            learning.do_episode_greedy()
            perform_cumreward = learning.last_rewardcumul
            last_cumulrewards.append(perform_cumreward)
            last_t.append(learning.last_time)
            update_wheel_trajectories()
        # perform_cumreward = learning.last_time
        # perform_cumrewards.append(perform_cumreward)
        perform_cumrewards.append(np.mean(last_cumulrewards))
        number_itterations.append(np.mean(last_t))
        episodes.append(learning.num_episode)

        ax1.cla()
        # ax1.plot(perform_cumrewards, '.--')
        ax1.plot(episodes, perform_cumrewards,'--')
        ax1.set_xlabel("Number of training episodes")
        ax1.set_ylabel("Performance")
        # Wheel trajectories

        ax3.cla()
        ax3.set_xlabel("Number of training episodes")
        ax3.set_ylabel("Number of itterations without falling /10")
        # ax3.set_xlim(0,10)
        ax3.plot(episodes, np.array(number_itterations)/10, '--')
        plt.pause(0.001)

# SAUVEGARDE
fig1.savefig("images_test/reward_"+str(params).replace("[", "").replace(":", "").replace("\n","").replace(" ", "")+".png")
fig2.savefig(
    "images_test/traject_" + str(params).replace("[", "").replace(":", "").replace("\n", "").replace(" ",
                                                                                                    "") + ".png")
fig3.savefig(
    "images_test/itterations_" + str(params).replace("[", "").replace(":", "").replace("\n", "").replace(" ",
                                                                                                    "") + ".png")