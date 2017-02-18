import numpy as np
import sys
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
import itertools


# Neural network for function approximation

class QLearning:
    def __init__(self, env, task, n_actions, alpha=1, gamma=0.99, epsilon=1, epsilon_decay=0.99):
        self.env = env
        self.task = task
        self.n_in = task.outdim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1

        self.lastaction = None
        self.laststate = None

        self.history = []

        # We create a model for each action
        self.models = []
        for _ in range(n_actions):
            # model = SGDRegressor(learning_rate="constant", penalty=None)
            model = MLPRegressor(hidden_layer_sizes=(100))
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            env.reset()
            model.partial_fit([self.task.getObservation()], [0])
            self.models.append(model)

        self.num_episode = 0

    def epsilon_greedy_policy(self, state, epsilon):
        is_random = np.random.binomial(1, epsilon)

        if is_random:
            return np.random.random_integers(0, self.n_actions-1)

        values = [self.models[a].predict([state])[0] for a in range(self.n_actions)]
        # print ("values : ", values)
        best_action = np.argmax(values)

        return best_action


    def do_episode(self):
        # One itteration of q learning
        self.env.reset()
        self.task.reset()

        reward_cumul = 0
        state = self.task.getObservation()
        for t in itertools.count():
            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)

            self.task.performAction(action)
            new_state = self.task.getObservation()

            reward = self.task.getReward()
            reward_cumul += self.gamma**t * reward

            q_values_next = [self.models[a].predict([new_state])[0] for a in range(self.n_actions)]

            td_target = reward + self.alpha * np.max(q_values_next)

            # print ("td target : ", td_target, " reward : ", reward)

            # We update our models
            self.models[action].partial_fit([state], [td_target])

            if self.task.isFinished():
                break

            state = new_state

        # print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t, "epsilon : ", self.epsilon)

        self.last_rewardcumul = reward_cumul
        self.last_time = t
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.num_episode += 1


    def do_episode_withoutLearning(self):
        # One itteration of q learning

        self.env.reset()
        self.task.reset()

        reward_cumul = 0
        state = self.task.getObservation()
        for t in itertools.count():
            action = self.epsilon_greedy_policy(state,epsilon=0.0)

            self.task.performAction(action)
            new_state = self.task.getObservation()

            reward = self.task.getReward()
            reward_cumul += self.gamma**t * reward

            if self.task.isFinished():
                break

            state = new_state

        print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t)

        self.last_rewardcumul = reward_cumul
        # self.num_episode += 1


