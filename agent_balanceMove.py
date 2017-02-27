import numpy as np
import itertools

class Agent_BalanceMove():

    def __init__(self, learnerBalance, learnerMove, CB=0.5, CM=0.5):
        self.learnerBalance = learnerBalance
        self.learnerMove = learnerMove
        self.task_Balance = learnerBalance.task
        self.env = self.task_Balance.env
        self.task_Move = learnerMove.task
        self.n_actions = learnerBalance.n_actions
        self.CB = CB
        self.CM = CM

    def  greedy_policy(self, state_balance, state_move):

        Q = self.CB * self.learnerBalance.Q[state_balance,:] + self.CM * self.learnerMove.Q[state_move, :]

        print ("QMove : ", np.sum(self.learnerMove.Q))
        print ("QBalance : ", np.sum(self.learnerBalance.Q))

        best_action = np.argmax(Q)
        return best_action


    def do_episode_greedy(self):

        # One itteration of q learning
        self.task_Balance.reset()
        self.task_Move.reset()
        self.env.reset()

        reward_cumul = 0

        num_state_Balance = np.argmax(self.task_Balance.getObservation())
        num_state_Move = np.argmax(self.task_Move.getObservation())

        action = self.greedy_policy(num_state_Balance, num_state_Move)
        for t in itertools.count():

            self.env._performAction(action)
            num_state_Balance = np.argmax(self.task_Balance.getObservation())
            num_state_Move = np.argmax(self.task_Move.getObservation())

            action = self.greedy_policy(num_state_Balance, num_state_Move)

            if self.task_Balance.isFinished() or self.task_Move.isFinished():
                break

        # print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t, "epsilon : ", self.epsilon)

        self.last_rewardcumul = reward_cumul
        self.last_time = t




