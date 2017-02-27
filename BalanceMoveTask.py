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
    return np.arccos(np.vdot(a, b) / (norm_a*norm_b))

def angle_between(p1, p2):
   ang1 = np.arctan2(*p1[::-1])
   ang2 = np.arctan2(*p2[::-1])
   tmp =  np.rad2deg((ang1 - ang2) % (2 * np.pi))
   return np.deg2rad((tmp + 180 + 360) % 360 - 180)


class BalanceMoveTask(BalanceTask):
    """The rider is to simply balance the bicycle while moving with the
    speed perscribed in the environment. This class uses a continuous 5
    dimensional state space, and a discrete state space.

    This class is heavily guided by
    pybrain.rl.environments.cartpole.balancetask.BalanceTask.

    """
    max_tilt = np.pi / 15
    nactions = 9

    def __init__(self, dest_x, dest_y, max_time=1000.0,env=None):
        BalanceTask.__init__(self,max_time,env)
        self.dest_x = dest_x
        self.dest_y = dest_y
        self.seuil_min = 5

    @property
    def indim(self):
        return 1

    @property
    def dest(self):
        return (self.dest_x, self.dest_y)

    @property
    def outdim(self):
        return len(self.getObservation())

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
         xf, yf, xb, yb, psi) = self.env.getSensors()

        return BalanceTask.getObservation(self) + [xf, yf, xb, yb]

    def isFinished(self):
        distance = dist(self.env.get_posf(), self.dest)

        if np.abs(self.env.getTilt()) > self.max_tilt:
            # print ("il est tombe")
            return True
        elapsed_time = self.env.time_step * self.t
        if elapsed_time > self.max_time:
            print ("> 1000")
            return True
        if distance < self.seuil_min:
            print ("pos f : ", self.env.get_posf(), " dest : ",self.dest)
            print ("Il est arrivee")
            return True
        return False

    def getReward(self, metrique=0):
        distance = dist(self.env.get_posf(), self.dest)
        dir_vect = np.subtract(self.dest,self.env.get_posf())
        angle = angle_vectors(self.env.get_posf(), self.dest)
        self.distance = distance

        if metrique == 0:
            if distance < self.seuil_min:
                return 5
            if np.abs(self.env.getTilt()) >= self.max_tilt:
                return -1
            return (4-angle**2)*0.0004
        elif metrique == 1:
            return 1/(distance*np.abs(angle))
        elif metrique == 2:
            if angle <= abs(self.last_angle):
                self.last_angle = angle
                return 5000
            else:
                self.last_angle = angle
                return 3000
        elif metrique == 3:
            if distance < self.seuil_min:
                return dist((0,0), self.dest)
            if np.abs(self.env.getTilt()) >= self.max_tilt:
                return -dist((0,0), self.dest)
            return -distance
        elif metrique == 4:
            if np.abs(self.env.getTilt()) >= self.max_tilt:
                return -1
            return 1/np.abs(angle)*0.01



class LinearFATileCodingBalanceMoveTask(BalanceMoveTask):
    dict = {}
    theta_bounds = np.array(
        [-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi])
    thetad_bounds = np.array(
        [-np.inf, -2.0, 0, 2.0, np.inf])
    omega_bounds = np.array(
        [-np.pi, -0.15, -0.06, 0, 0.06, 0.15,
         np.pi])
    omegad_bounds = np.array(
        [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
    omegadd_bounds = np.array(
        [-np.inf, -2.0, 0, 2.0, np.inf])

    # Angle entre la roue avant et la destination
    # psig_bounds = np.array([-np.pi,-5*np.pi/6, -2*np.pi/3, -np.pi/2, -np.pi/3, -np.pi/6, 0,
    #                         np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, np.pi], dtype=np.float)

    psig_bounds = np.array(np.linspace(-np.pi, np.pi, 18))



    # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
    nbins_across_dims = [
        len(theta_bounds) - 1,
        len(thetad_bounds) - 1,
        len(omega_bounds) - 1,
        len(omegad_bounds) - 1,
        len(omegadd_bounds) - 1,
        len(psig_bounds)-1]

    i = 0
    # for t in range(nbins_across_dims[0]):
    #     for td in range(nbins_across_dims[1]):
    #         for o in range(nbins_across_dims[2]):
    #             for od in range(nbins_across_dims[3]):
    #                 for odd in range(nbins_across_dims[4]):
    #                     for pg in range(nbins_across_dims[5]):
    #                         dict[(t, td, o, od, odd,pg)] = i
    #                         i += 1

    for t in range(nbins_across_dims[0]):
        for td in range(nbins_across_dims[1]):
            for o in range(nbins_across_dims[2]):
                for od in range(nbins_across_dims[3]):
                    for odd in range(nbins_across_dims[4]):
                        dict[(t, td, o, od, odd)] = i
                        i += 1

    dimension = i

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
         xf, yf, xb, yb, psi) = self.env.getSensors()
        # psig = angle_vectors(np.subtract(self.env.get_posf(),self.env.get_posb()), self.dest)
        psig = angle_vectors(self.env.get_posf(), self.dest)
        # psig = angle_vectors(self.env.get_posf(), self.env.get_posb())
        bin_indices = [
            self.get_num_interval(theta, self.theta_bounds),
            self.get_num_interval(thetad, self.thetad_bounds),
            self.get_num_interval(omega, self.omega_bounds),
            self.get_num_interval(omegad, self.omegad_bounds),
            self.get_num_interval(omegadd, self.omegadd_bounds),
            # self.get_num_interval(psig, self.psig_bounds)
        ]

        indice_interval = bin_indices[-1]

        # print ("(xf,yf) : ", self.env.get_posf(), " dest : ", self.dest)
        self.value_angle = (self.psig_bounds[indice_interval] + self.psig_bounds[indice_interval+1])/2.

        num_state = self.dict[tuple(bin_indices)]
        state = np.zeros((self.dimension))
        state[num_state] = 1
        return state

    def getReward(self, metrique=0):
        # -1 reward for falling over; no reward otherwise.

        distance = dist(self.env.get_posf(), self.dest)
        dir_vect = np.subtract(self.dest, self.env.get_posf())
        angle = angle_vectors(np.subtract(self.env.get_posf(),self.env.get_posb()), self.dest)
        # angle = angle_between(self.env.get_posf(), self.dest)

        self.distance = distance

        if metrique == 0:
            if distance < self.seuil_min:
                return 1
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -1
            return (4 - angle ** 2) * 0.0004
            # return 0
        elif metrique == 1:
            return 1 / (distance * np.abs(angle))
        elif metrique == 2:
            if angle <= abs(self.last_angle):
                self.last_angle = angle
                return 5000
            else:
                self.last_angle = angle
                return 3000
        elif metrique == 3:
            if distance < self.seuil_min:
                return dist((0, 0), self.dest)
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -dist((0, 0), self.dest)
            return -0.001*distance
        elif metrique == 4:
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -1
            return 1 / np.abs(angle) * 0.01
        elif metrique==5:
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -1
            return 1/float(distance)


    def get_num_interval(self, value, bound):
        for i in range(len(bound) - 1):
            if value >= bound[i] and value < bound[i + 1]:
                return i
        return i

class LinearFATileCodingMoveTask(BalanceMoveTask):
    dict = {}

    # Angle entre la roue avant et la destination
    psig_bounds = np.array(np.linspace(-np.pi, np.pi, 20))
    max_tilt = np.pi

    nbins_across_dims = [
        len(psig_bounds) - 1]

    i = 0
    for pg in range(nbins_across_dims[0]):
        dict[(pg,)] = i
        i += 1

    dimension = i
    print ("dict : ", dict)


    def getObservation(self):
        self.psig = angle_vectors(np.subtract(self.env.get_posf(),self.env.get_posb()), self.dest)
        bin_indices = [
            self.get_num_interval(self.psig, self.psig_bounds)
        ]

        indice_interval = bin_indices[-1]

        # print ("(xf,yf) : ", self.env.get_posf(), " dest : ", self.dest)
        self.value_angle = (self.psig_bounds[indice_interval] + self.psig_bounds[indice_interval + 1]) / 2.

        num_state = self.dict[tuple(bin_indices)]
        state = np.zeros((self.dimension))
        state[num_state] = 1
        return state

    def isFinished(self):
        elapsed_time = self.env.time_step * self.t
        distance = dist(self.env.get_posf(), self.dest)

        if np.abs(self.env.getTilt()) > self.max_tilt:
            # print ("il est tombe")
            return True
        if elapsed_time > self.max_time:
            print ("> 1000")
            return True
        if distance < self.seuil_min:
            print ("pos f : ", self.env.get_posf(), " dest : ", self.dest)
            print ("Il est arrivee")
            return True
        if distance > 100 * self.seuil_min:
            return True
        return False

    def getReward(self, metrique=0):
        distance = dist(self.env.get_posf(), self.dest)
        if metrique == 0:
            if distance < self.seuil_min:
                return 1
            if distance > 100*self.seuil_min:
                return -1
            return (4 - self.psig ** 2) * 0.004
        elif metrique==1:
            if distance < self.seuil_min:
                np.pi
            return -np.abs(self.psig)

    def get_num_interval(self, value, bound):
        for i in range(len(bound) - 1):
            if value >= bound[i] and value < bound[i + 1]:
                return i
        return i

