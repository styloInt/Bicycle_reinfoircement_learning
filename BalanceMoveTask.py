import pybrain
import numpy as np
from bicycle_env import BicycleEnvironment
from pybrain.utilities import one_to_n
from scipy.spatial import distance
from BalanceTask import BalanceTask

def dist(a, b):
    return (distance.euclidean(a,b))

def angle_vectors(a ,b):
    xa, ya = a; xb,yb = b
    norm_a = np.sqrt(xa**2+ya**2)
    norm_b = np.sqrt(xb**2+yb**2)

    return np.arccos(np.vdot((xa,xb), (xb,yb)) / (norm_a*norm_b))


class BalanceMoveTask(BalanceTask):
    """The rider is to simply balance the bicycle while moving with the
    speed perscribed in the environment. This class uses a continuous 5
    dimensional state space, and a discrete state space.

    This class is heavily guided by
    pybrain.rl.environments.cartpole.balancetask.BalanceTask.

    """
    max_tilt = np.pi / 6.
    nactions = 9

    def __init__(self, dest_x, dest_y, max_time=1000.0):
        BalanceTask.__init__(self)
        self.dest_x = dest_x
        self.dest_y = dest_y
        self.seuil_min = 0.5

    @property
    def indim(self):
        return 1

    @property
    def dest(self):
        return (self.dest_x, self.dest_y)

    @property
    def outdim(self):
        return 10

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
         xf, yf, xb, yb, psi) = self.env.getSensors()
        return BalanceTask.getObservation(self) + [xf, yf, xb, yb, psi]

    def isFinished(self):
        if np.abs(self.env.getTilt()) > self.max_tilt:
            # print ("il est tombe")
            return True
        elapsed_time = self.env.time_step * self.t
        if elapsed_time > self.max_time:
            print ("> 1000")
            return True
        if dist(self.env.get_posf(), self.dest) < self.seuil_min:
            print ("pos f : ", self.env.get_posf(), " dest : ",self.dest)
            print ("Il est arrivee")
            return True
        return False

    def getReward(self):
        # -1 reward for falling over; no reward otherwise.

        distance = dist(self.env.get_posf(), self.dest)
        dir_vect = np.subtract(self.dest,self.env.get_posf())
        angle = angle_vectors(self.env.get_posf(), dir_vect)

        reward = 0
        #
        if distance < self.seuil_min:
            return 1

        if np.abs(self.env.getTilt()) > self.max_tilt:
            reward += -1
            return reward
        #
        # return (4-abs(self.env.getTilt())**2)*0.00004

        return -abs(self.env.getSteer()) / (np.pi*self.dest_y)

        # self.B = 16
        # reward += -self.B * abs(self.env.getTilt()) * distance
        # return reward


