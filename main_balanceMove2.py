import numpy as np
from matplotlib import pyplot as plt

from QLearning import QLearning
from Sarsa_lambda import Sarsa_lambda
from BalanceTask import BalanceTask, LinearFATileCodingBalanceTask
import itertools
from BalanceMoveTask import BalanceMoveTask, LinearFATileCodingBalanceMoveTask, LinearFATileCodingMoveTask
from agent_balanceMove import Agent_BalanceMove
import seaborn

"""
The goal here is to combine two different learning.
One for learning how to balance and another one who learn the best way to go to a point.
"""

# task = BalanceMoveTask(0,10)
task_balance = LinearFATileCodingBalanceTask()
env = task_balance.env
task_move = LinearFATileCodingMoveTask(0,100, env=env)

# learning = QLearning(env, task, 9, alpha=0.05, epsilon_min=0.0, gamma=0.99, epsilon_decay=0.999, lambd=0.7,Sarsa=True)
learning_balance = Sarsa_lambda(env, task_balance, 9, alpha=0.05, epsilon_min=0.0,epsilon=1, gamma=0.9, epsilon_decay=0.999, lambd=0.7, metrique_reward=0)
learning_move = Sarsa_lambda(env, task_move, 9, alpha=0.05, epsilon_min=0.0,epsilon=1, gamma=0.9, epsilon_decay=0.999, lambd=0.7, metrique_reward=0)
agent_balanceMove = Agent_BalanceMove(learnerBalance=learning_balance, learnerMove=learning_move, CB=0.4, CM=0.6)

env.saveWheelContactTrajectories(True)
plt.ion()
# plt.figure(figsize=(8, 4))

# fig1 = plt.figure()
# ax1 = plt.axes()
fig2 = plt.figure()
ax2 = plt.axes()
# fig3 = plt.figure()
# ax3 = fig3.gca()

def update_wheel_trajectories():
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')

params = ["alpha : ", learning_balance.alpha, "K_disconout_factor : ", learning_balance.K_discountFactor, "epsilon_min : ", learning_balance.epsilon_min, "gamma : ", learning_balance.gamma,\
                                       " epsilon_decay : ", learning_balance.epsilon_decay, "lambda : ", learning_balance.lambd, " metrique : ", learning_balance.metrique_reward, "dest : ", task_move.dest]

print (params)

perform_cumrewards = []
distances = []
episodes = []
number_episodes = 10000
for irehearsal in itertools.count():
    if learning_balance.num_episode > number_episodes:
        break

    # Learn.
    # ------

    learning_balance.do_episode()
    learning_move.do_episode()
    if irehearsal % 10 == 0:
        # Perform (no learning).
        # ----------------------
        # Perform.
        last_cumulrewards = []
        last_distance = []
        for t in range(3):
            agent_balanceMove.do_episode_greedy()
            # perform_cumreward = learning.last_rewardcumul
            # last_cumulrewards.append(perform_cumreward)
            # last_distance.append(task.distance)
            update_wheel_trajectories()
        # perform_cumreward = learning.last_time
        # perform_cumrewards.append(perform_cumreward)
        # perform_cumrewards.append(np.mean(last_cumulrewards))
        # distances.append(np.mean(last_distance))
        # episodes.append(learning.num_episode)

        # ax1.cla()
        # # ax1.plot(perform_cumrewards, '.--')
        # ax1.plot(episodes, perform_cumrewards,'--')
        # ax1.set_xlabel("Number of training episodes")
        # ax1.set_ylabel("Performance")
        # # Wheel trajectories
        #
        # ax3.cla()
        # ax3.set_xlabel("Number of training episodes")
        # ax3.set_ylabel("Distance from the goal")
        # # ax3.set_xlim(0,10)
        # ax3.plot(episodes, np.array(distances), '--')
        plt.pause(0.001)

fig1.savefig("images_test_Move/reward_"+str(params).replace("[", "").replace(":", "").replace("\n","").replace(" ", "")+".png")
fig2.savefig(
    "images_test_Move/traject_" + str(params).replace("[", "").replace(":", "").replace("\n", "").replace(" ",
                                                                                                    "") + ".png")
fig3.savefig(
    "images_test_Move/distance_" + str(params).replace("[", "").replace(":", "").replace("\n", "").replace(" ",
                                                                                                    "") + ".png")