# coding: utf-8

# In[1]:

import itertools
import matplotlib
import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing
from matplotlib import pyplot as plt
from copy import deepcopy
#get_ipython().magic(u'matplotlib qt')


# In[ ]:

#import the environnement
from bicycle_env import BicycleEnvironment
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPRegressor


# In[ ]:

env = BicycleEnvironment()


# In[ ]:

#basic constandts :
model_name = "nn"
nb_actions = 9
nb_states = 10 #for the begguging


env.reset()

#discretisation
#------------------------------------------------
angle_bar =[-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi]
angular_vecolity = [-np.inf,-2.0,0,2.0,np.inf]
angle_vertical = [-np.pi / 6., -0.15, -0.06, 0, 0.06, 0.15,np.pi / 6.]
angular_vecolity_d = [-np.inf,-0.5,-0.25,0,0.25,0.5,np.inf]
angular_accel = [-np.inf,-2.0,0,2.0,np.inf]
entier = [angle_bar,angular_vecolity,angle_vertical,angular_vecolity_d,angular_accel]
state_conversion = []
to_int = [ 1 ,6,24,144,864]

def discrtisation_etats(etats):
    arr = []
    for i in range(len(etats)):
        arr.append(np.digitize([etats[i]], entier[i]))
    arr = np.array(arr)
    arr -= 1
    val =  np.array(arr.reshape(1,-1)[0]).astype(int)
    pred = np.sum([val[i] * to_int[i] for i in range(len(val))])
    #tab =  np.zeros(3456)
    #tab[pred]=1
    return pred


# In[ ]:

#metric choice :
metric_id = 1
max_tilt = np.pi / 15.
def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    tmp =  np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return np.deg2rad((tmp + 180 + 360) % 360 - 180)
def get_reward(state):
    if metric_id ==0:
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        return 0.0
    elif metric_id ==1:
        #chute du velo
        if np.abs(env.getTilt()) > max_tilt:
            return -1.0
        #cible atteinte
        elif state[6]>100:
            return 0.001
        else:
            #rien de special
            tmp = angle_between((state[5],state[6]),(0,100))
            return (4-np.power(tmp,2))*0.00004

def int_to_act(number_action):
    a1 = (number_action/3)
    a2 = (number_action%3)
    return (a1,a2)
def act_to_int(act):
    return act[0] *3 +act[1]
def epsilon_greedy_policy(estimated_reward,epsilon):
    if  np.random.random() > epsilon:
        #exploitation
        best = np.argmax(estimated_reward)
        return int_to_act(best)
    else:
        #exploration :
        return np.random.randint(3,size=2)
def update_wheel_trajectories(ax2):
    #code from the pybrain example
    front_lines = ax2.plot(env.get_xfhist(), env.get_yfhist(), 'r')
    back_lines = ax2.plot(env.get_xbhist(), env.get_ybhist(), 'b')
    plt.axis('equal')
def to_action(action):
    bar_t = [-2.0,0,2.0]#[-1.0,0,1.0]
    speed = [-0.02,0,0.02]
    return [bar_t[action[0]],bar_t[action[1]]]
def sarsa_lambda(env,graphics = False,nb_episodes=10):
    discount_factor = 0.99
    rewards_cumules = []
    if graphics:
        env.saveWheelContactTrajectories(True)
        plt.ion()
        fif = plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
    should_show = False
    nb_actions = 9
    nb_states = 3456
    #contient les q-values
    q_values = np.zeros((nb_states,nb_actions))
    K = 2000
    target_reacehed = False
    for ep in range(nb_episodes):
        #reset the environnment
        env.reset()
        #initialize the traces:
        traces  = np.zeros((nb_states,nb_actions))
        #next action
        a_next = np.random.randint(3,size=2)
        #time to learning
        t = 0
        reward_cuml = 0
        epsilon = 1.0
        #current state
        current_state = discrtisation_etats(env.sensors[:5])
        if (ep+1) % 50 == 0:
            should_show = True
            epsilon = 0.0
        else:
            should_show = False
            epsilon = 1.0
        if target_reacehed:
            epsilon = 0.0
        test = {}
        learning_rate = 0.5
        while True:
            if env.time_step>0:
                if a_next != None:
                    action = a_next
                #make the action
                env.performAction(to_action(action))
                obser = discrtisation_etats(env.sensors[:5])
                reward = get_reward(env.sensors)
                #estimation
                next_values = q_values[obser]
                #print(next_values)
                #next action
                a_next = epsilon_greedy_policy(next_values,epsilon)
                if act_to_int(a_next) in test:
                    test[act_to_int(a_next)] +=1
                else:
                    test[act_to_int(a_next)] = 1
                #print("new action : ", a_n ext,epsilon)
                estimated_value =  reward + discount_factor * next_values[act_to_int(a_next)] - q_values[current_state][act_to_int(action)]
                traces[current_state][act_to_int(action)] += 1
                #approx.update(current_state, action, estimated_value)
                #print(estimated_value,reward , discount_factor ,next_values[act_to_int(a_next)],q_values[current_state][act_to_int(action)],  discount_factor * next_values[act_to_int(a_next)] - q_values[current_state][act_to_int(action)])
                #q_values[current_state][act_to_int(a_next)] =  q_values[current_state][act_to_int(a_next)] + learning_rate* estimated_value
                '''for s in range(len(traces)):
                    for a in  range(len(traces[s])):
                        q_values[s][a] = q_values[s][a] +  0.5*estimated_value*traces[s][a]
                        traces[s][a] = traces[s][a] * 0.99* 0.95'''
                if not target_reacehed:
                    q_values = q_values+ learning_rate*estimated_value*traces
                    traces = traces * 0.99* 0.95
                reward_cuml += reward * np.power(1.0,t)
                epsilon *= 0.985
                #epsilon = max(epsilon,0.15)
                current_state = obser
                action = a_next
            if reward == -1.0 or reward==0.001:
                if reward == 0.001:
                    target_reacehed = True
                learning_rate = (2000)/float((2000+ep))
                #print("Distance parcourue jusqu'a chute  : " , env.sensors[6] , t)
                #row_sums = q_values.sum(axis=1)+10**-7
                #q_values = q_values / row_sums[:, np.newaxis]
                if should_show:
                    rewards_cumules.append(reward_cuml)
                if False:
                    approx.retrain_scaler()
                break
            t += 1
        if graphics and should_show:
            print(env.sensors[6])
            ax1.cla()
            ax1.plot(rewards_cumules, '.--')
            update_wheel_trajectories(ax2)
            plt.pause(0.001)
sarsa_lambda(env,nb_episodes = 400000, graphics=True)
'''print("Nombre d'états : ",(len(angle_bar)-1)*(len(angular_vecolity)-1)*(len(angle_vertical)-1)* (len(angular_vecolity_d)-1)*(len(angular_accel)-1))
indices_discretsation = []
print(discrtisation_etats([-1.1,0,0,0,0]))


test = np.cumprod([1] + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])[:-1]
print([1]  + [len([1,2,3,4,6,7])-1,len([5,6,3,2])-1, len([1,2,3])-1,len([4,5,6,4])-1])
print("Test : ",test,len([1,2,3,4,6,7]))
'''
