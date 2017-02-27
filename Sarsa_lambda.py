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

class Sarsa_lambda:
    def __init__(self, env, task, n_actions, alpha=None, gamma=0.99, epsilon=1, epsilon_decay=0.9, K_discountFactor=None, epsilon_min=0.05, lambd=0.9, metrique_reward=0):
        assert (alpha != None and K_discountFactor == None) or (alpha==None and K_discountFactor!=None)

        # self.env = env
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
        self.lambd = lambd
        self.metrique_reward = metrique_reward

        self.state_nonvisited = set(range(self.n_in))

        self.lastaction = None
        self.laststate = None

        self.history = []

        # We create a model for each action
        self.models = []
        env.reset()
        for _ in range(n_actions):
            # model = SGDRegressor()
            model = MLPRegressor(hidden_layer_sizes=())
            model.partial_fit([self.task.getObservation()], [0])
            self.models.append(model)

        self.num_episode = 0

        self.traces = np.zeros((self.task.outdim, n_actions))
        self.Q = np.zeros((self.task.outdim, n_actions))
        # self.Q = np.random.rand(self.task.outdim, n_actions)

    def epsilon_greedy_policy(self, state, epsilon):
        is_random = np.random.binomial(1, epsilon)

        if is_random:
            return np.random.random_integers(0, self.n_actions-1)
        # print ("values : ", values)
        best_action = np.argmax(self.Q[state,:])

        return best_action

    def _boltzmannProbs(self,qvalues, temperature=1.):
        if temperature == 0:
            tmp = np.zeros(len(qvalues))
            tmp[np.r_argmax(qvalues)] = 1.
        else:
            tmp = qvalues / temperature
            tmp -= max(tmp)
            tmp = np.exp(np.clip(tmp, -20, 0))
        return tmp / sum(tmp)

    def boltzman_policy(self,state):
        self.Q[state, :] = softmax(self.Q[state,:])
        probs = self._boltzmannProbs(self.Q[state,:], temperature=1)
        best_action = np.random.choice(np.arange(len(probs)), p=probs)

        return best_action


    def do_episode(self, epsilon=None):
        # One itteration of q learning
        self.task.reset()
        self.traces = np.zeros(self.traces.shape)
        learning = True
        if epsilon == None:
            epsilon=self.epsilon
        else:
            learning = False

        # self.epsilon = self.epsilon_init/float(self.epsilon_init+self.num_episode)

        # self.epsilon = self.epsilon_init

        if self.K_discountFactor:
            self.alpha = (self.K_discountFactor / float(self.K_discountFactor + self.num_episode))

        reward_cumul = 0
        state = self.task.getObservation()
        num_state = np.argmax(state)
        # if epsilon != None:
        #     action = self.epsilon_greedy_policy(num_state, epsilon=epsilon)
        # else:
        #     action = self.boltzman_policy(num_state)
        action = self.epsilon_greedy_policy(num_state, epsilon=epsilon)
        for t in itertools.count():

            self.task.performAction(action)
            new_state = self.task.getObservation()

            reward = self.task.getReward(self.metrique_reward)
            reward_cumul += 0.99**t*reward

            num_newState = np.argmax(new_state)
            # try:
            #     self.state_nonvisited.remove(num_newState)
            # except KeyError:
            #     print
            # print ("state non visited : ", len(self.state_nonvisited))
            #
            # if epsilon != None:
            #     new_action = self.epsilon_greedy_policy(num_newState, epsilon=epsilon)
            # else:
            #     new_action = self.boltzman_policy(num_newState)

            new_action = self.epsilon_greedy_policy(num_newState, epsilon=epsilon)

            # print ("reward : ", reward)
            if learning:
                delta = reward + self.gamma*self.Q[num_newState, new_action] - self.Q[num_state, action]
                # print ("action : ", action)
                self.traces[num_state, action] += 1
                self.Q += self.alpha*delta*self.traces
                self.traces *= self.gamma * self.lambd

            if self.task.isFinished():
                break

            state = new_state
            num_state=num_newState
            action = new_action
            # epsilon *= self.epsilon_decay

        # print ("Episode : ", self.num_episode, " , cummul reward ", reward_cumul, "time : ", t, "epsilon : ", self.epsilon)

        self.last_rewardcumul = reward_cumul
        self.last_time = t

        if epsilon != 0.0 and self.epsilon >= self.epsilon_min and learning:
            self.epsilon =epsilon*self.epsilon_decay

        if learning:
            self.num_episode += 1


    # Always choose the best action : epsilon = 0
    def do_episode_greedy(self):
        self.do_episode(epsilon=0.0)
        print ("Episode : ", self.num_episode, " , cummul reward ", self.last_rewardcumul, "time : ", self.last_time, "alpha : ", self.alpha, " epsilon : ", self.epsilon)

    def samples_state(self, n):

        samples = []
        for i in range(0,n):
            state = self.task.getObservation()
            action = self.epsilon_greedy_policy(state, epsilon=1.0)
            self.task.performAction(action)
            samples.append(state)

            if self.task.isFinished():
                self.task.reset()

        return samples


