import pybrain
import numpy as np
from bicycle_env import BicycleEnvironment
from pybrain.utilities import one_to_n


class BalanceTask(pybrain.rl.environments.EpisodicTask):
    """The rider is to simply balance the bicycle while moving with the
    speed perscribed in the environment. This class uses a continuous 5
    dimensional state space, and a discrete state space.

    This class is heavily guided by
    pybrain.rl.environments.cartpole.balancetask.BalanceTask.

    """
    max_tilt = np.pi / 15.
    nactions = 9

    def __init__(self, max_time=1000.0, env=None):
        if env==None:
            env = BicycleEnvironment()
        super(BalanceTask, self).__init__(env)
        self.max_time = max_time
        # Keep track of time in case we want to end episodes based on number of
        # time steps.
        self.t = 0

        self.lastT = 0
        self.lastd = 0
        self.last_action=0
        self.last_angle = 0

    @property
    def indim(self):
        return 1

    @property
    def outdim(self):
        # return 5
        return len(self.getObservation())

    def reset(self):
        super(BalanceTask, self).reset()
        self.env.reset()
        self.lastT = 0
        self.lastd = 0
        self.last_action=0
        self.last_angle = 0
        self.t = 0

    def performAction(self, action):
        """Incoming action is an int between 0 and 8. The action we provide to
        the environment consists of a torque T in {-2 N, 0, 2 N}, and a
        displacement d in {-.02 m, 0, 0.02 m}.

        """
        self.t += 1
        assert round(action) == action

        # -1 for action in {0, 1, 2}, 0 for action in {3, 4, 5}, 1 for
        # action in {6, 7, 8}
        torque_selector = np.floor(action / 3.0) - 1.0
        T = 2 * torque_selector
        # Random number in [-1, 1]:
        p = 2.0 * np.random.rand() - 1.0
        # -1 for action in {0, 3, 6}, 0 for action in {1, 4, 7}, 1 for
        # action in {2, 5, 8}
        disp_selector = action % 3 - 1.0
        d = 0.02 * disp_selector + 0.02 * p
        super(BalanceTask, self).performAction([T, d])

        self.lastT = T
        self.lastd = d
        self.last_action = action

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
         xf, yf, xb, yb, psi) = self.env.getSensors()
        X = self.env.getSensors()
        x_2 = np.power(X[2], 2)
        x_0 = np.power(X[0], 2)
        res = [1, X[2], X[3], x_2, np.power(X[3], 2), X[2] * X[3], X[0], X[1], x_0, np.power(X[1], 2),
               X[0] * X[1], X[2] * X[0], X[2] * x_0, x_2 * X[0]]

        return list(self.env.getSensors()[0:5]) + [self.last_action]
        # return res

    def isFinished(self):
        # Criterion for ending an episode. From Randlov's paper:
        # "When the agent can balance for 1000 seconds, the task is considered
        # learned."
        if np.abs(self.env.getTilt()) > self.max_tilt:
            # print ("il est tombe")
            return True
        elapsed_time = self.env.time_step * self.t
        if elapsed_time > self.max_time:
            print ("> 1000")
            return True
        return False

    def getReward(self, metrique=0):
        # Decomentez la reward que vous voulez utiliser, et commentez le reste
        if metrique==0:
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -1.
            return 0
        elif metrique==1:
            if np.abs(self.env.getTilt()) > self.max_tilt:
                return -1.
            return 0.01
        elif metrique==2:
            return self.getReward_angle()



    # reward qui se base sur l'angle d'inclinaison du velo
    def getReward_angle(self):
        # diff = abs(self.last_angle) - abs(self.env.getTilt())*10
        # return diff

        if abs(self.env.getTilt()) <= abs(self.last_angle):
            self.last_angle = self.env.getTilt()
            return 5000
        else:
            self.last_angle = self.env.getTilt()
            return 3000

    # -1 reward for falling over; no reward otherwise.
    def getReward_fail(self):
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return -10.
        return 0



class LinearFATileCodingBalanceTask(BalanceTask):
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



    # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
    nbins_across_dims = [
            len(theta_bounds) - 1,
            len(thetad_bounds) - 1,
            len(omega_bounds) - 1,
            len(omegad_bounds) - 1,
            len(omegadd_bounds) - 1]

    i=0
    for t in range(nbins_across_dims[0]):
        for td in range(nbins_across_dims[1]):
            for o in range(nbins_across_dims[2]):
                for od in range(nbins_across_dims[3]):
                    for odd in range(nbins_across_dims[4]):
                        dict[(t,td,o,od,odd)] = i
                        i+=1

    def getObservation(self):
        (theta, thetad, omega, omegad, omegadd,
                xf, yf, xb, yb, psi) = self.env.getSensors()
        bin_indices = [
            self.get_num_interval(theta, self.theta_bounds),
            self.get_num_interval(thetad, self.thetad_bounds),
            self.get_num_interval(omega, self.omega_bounds),
            self.get_num_interval(omegad, self.omegad_bounds),
            self.get_num_interval(omegadd, self.omegadd_bounds),
        ]

        num_state = self.dict[tuple(bin_indices)]
        state = np.zeros((3456))
        state[num_state] = 1
        return state

    def get_num_interval(self, value, bound):
        for i in range(len(bound)-1):
            if value >= bound[i] and value < bound[i+1]:
                return i
        print ("value : ", value)
        print ("bound : ", bound)
        # return i
