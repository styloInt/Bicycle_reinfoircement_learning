import numpy as np
import sys
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_approximation import RBFSampler
import itertools
import sklearn.pipeline

import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Neural network for function approximation

class QLearning:
    def __init__(self, env, task, n_actions, alpha=None, gamma=0.99, epsilon=1, epsilon_decay=0.99, K_discountFactor=None, epsilon_min=0.05, Sarsa=True, lambd=0.5):
        assert (alpha != None and K_discountFactor == None) or (alpha==None and K_discountFactor!=None)

        self.env = env
        self.task = task
        self.n_in = task.outdim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_init = 1000
        self.K_discountFactor = K_discountFactor
        self.Sarsa=Sarsa
        self.lambd = lambd

        self.lastaction = None
        self.laststate = None

        self.history = []

        # We create a model for each action
        self.models = []
        env.reset()
        for _ in range(n_actions):
            model = SGDRegressor()
            # model = MLPRegressor(hidden_layer_sizes=())
            model.partial_fit([self.task.getObservation()], [0])
            self.models.append(model)

        self.num_episode = 0

        # We random states
        observation_examples = self.samples_state(500)
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        self.scaler.transform(observation_examples)

    def epsilon_greedy_policy(self, state, epsilon):
        np.random.seed()
        is_random = np.random.binomial(1, epsilon)

        if is_random:
            return np.random.random_integers(0, self.n_actions-1)

        values = [self.models[a].predict([self.featurize_state(state)])[0] for a in range(self.n_actions)]
        # print ("values : ", values)
        best_action = np.argmax(values)

        return best_action


    def do_episode(self):
        # One itteration of q learning
        self.env.reset()
        self.task.reset()

        # self.epsilon = self.epsilon_init/float(self.epsilon_init+self.num_episode)

        # self.epsilon = self.epsilon_init

        if self.K_discountFactor:
            self.alpha = self.K_discountFactor /float(self.K_discountFactor+self.num_episode)

        reward_cumul = 0
        state = self.task.getObservation()
        if self.Sarsa:
            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
        for t in itertools.count():
            if not self.Sarsa:
                action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)


            self.task.performAction(action)
            new_state = self.task.getObservation()

            reward = self.task.getReward()
            reward_cumul += self.gamma**t * reward

            q_values_next = [self.models[a].predict([self.featurize_state(new_state)])[0] for a in range(self.n_actions)]


            if not self.Sarsa:
                td_target =  reward + self.alpha * np.max(q_values_next)
            else:
                # proba = softmax(q_values_next)
                # action = np.random.choice(np.arange(len(proba)), p=proba)
                action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
                td_target = reward + self.alpha * q_values_next[action]

            # print ("td target : ", td_target, " reward : ", reward)

            # We update our models
            self.models[action].partial_fit([self.featurize_state(state)], [td_target])

            if self.task.isFinished():
                break

            state = new_state
            # self.epsilon *= self.epsilon_decay

        # print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t, "epsilon : ", self.epsilon)

        self.last_rewardcumul = reward_cumul
        self.last_time = t

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.num_episode += 1


    # Always choose the best action : epsilon = 0
    def do_episode_withoutLearning(self):
        # One itteration of q learning

        self.env.reset()
        self.task.reset()

        reward_cumul = 0

        state = self.task.getObservation()
        if self.Sarsa:
            action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
        for t in itertools.count():
            if not self.Sarsa:
                action = self.epsilon_greedy_policy(state, epsilon=0)

            self.task.performAction(action)
            new_state = self.task.getObservation()

            reward = self.task.getReward()
            reward_cumul += self.gamma ** t * reward

            q_values_next = [self.models[a].predict([self.featurize_state(new_state)])[0] for a in
                             range(self.n_actions)]

            if not self.Sarsa:
                td_target = reward + self.alpha * np.max(q_values_next)
            else:
                action = self.epsilon_greedy_policy(state, epsilon=self.epsilon)
                td_target = reward + self.alpha * q_values_next[action]


            # We update our models
            self.models[action].partial_fit([self.featurize_state(state)], [td_target])

            if self.task.isFinished():
                break

            state = new_state

        print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t, "alpha : ", self.alpha, " epsilon : ", self.epsilon)

        self.last_rewardcumul = reward_cumul
        self.num_episode += 1


    def samples_state(self, n):

        samples = []
        for i in range(0,n):
            state = self.task.getObservation()
            action = self.epsilon_greedy_policy(state, epsilon=1.0)
            self.task.performAction(action)
            samples.append(state)

            if self.task.isFinished():
                self.env.reset()
                self.task.reset()

        return samples

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        # return state
        scaled = self.scaler.transform(np.array(state).reshape(1,-1))
        return scaled[0]


