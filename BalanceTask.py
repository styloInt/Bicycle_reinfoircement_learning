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
    max_tilt = np.pi / 6.
    nactions = 9

    def __init__(self, max_time=1000.0):
        super(BalanceTask, self).__init__(BicycleEnvironment())
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
        return 5+1 # We keep track of the last action

    def reset(self):
        super(BalanceTask, self).reset()
        self.lastT = 0
        self.lastd = 0
        self.last_action=0
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
        return list(self.env.getSensors()[0:5]) + [self.last_action]

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

    def getReward(self):
        # Decomentez la reward que vous voulez utiliser, et commentez le reste
        # return self.getReward_angle()
        return self.getReward_fail()


    # reward qui se base sur l'angle d'inclinaison du velo
    def getReward_angle(self):
        diff = abs(self.last_angle) - abs(self.env.getTilt())*10
        return diff

        # print ("angle : ", self.env.getTilt())
        # if abs(self.env.getTilt()) <= abs(self.last_angle):
        #     self.last_angle = self.env.getTilt()
        #     return 1
        # else:
        #     self.last_angle = self.env.getTilt()
        #     return -1

    # -1 reward for falling over; no reward otherwise.
    def getReward_fail(self):
        if np.abs(self.env.getTilt()) > self.max_tilt:
            return 0
        return 0.1



# class LinearFATileCodingBalanceTask(BalanceTask):
#     theta_bounds = np.array(
#             [-0.5 * np.pi, -1.0, -0.2, 0, 0.2, 1.0, 0.5 * np.pi])
#     thetad_bounds = np.array(
#             [-np.inf, -2.0, 0, 2.0, np.inf])
#     omega_bounds = np.array(
#             [-BalanceTask.max_tilt, -0.15, -0.06, 0, 0.06, 0.15,
#                 BalanceTask.max_tilt])
#     omegad_bounds = np.array(
#             [-np.inf, -0.5, -0.25, 0, 0.25, 0.5, np.inf])
#     omegadd_bounds = np.array(
#             [-np.inf, -2.0, 0, 2.0, np.inf])
#     # http://stackoverflow.com/questions/3257619/numpy-interconversion-between-multidimensional-and-linear-indexing
#     nbins_across_dims = [
#             len(theta_bounds) - 1,
#             len(thetad_bounds) - 1,
#             len(omega_bounds) - 1,
#             len(omegad_bounds) - 1,
#             len(omegadd_bounds) - 1]
#     # This array, when dotted with the 5-dim state vector, gives a 'linear'
#     # index between 0 and 3455.
#     magic_array = np.cumprod([1] + nbins_across_dims)[:-1]
#
#
#     @property
#     def outdim(self):
#         # Used when constructing LinearFALearner's.
#         return 3456
#
#     def getBin(self, theta, thetad, omega, omegad, omegadd):
#         bin_indices = [
#                 np.digitize([theta], self.theta_bounds)[0] - 1,
#                 np.digitize([thetad], self.thetad_bounds)[0] - 1,
#                 np.digitize([omega], self.omega_bounds)[0] - 1,
#                 np.digitize([omegad], self.omegad_bounds)[0] - 1,
#                 np.digitize([omegadd], self.omegadd_bounds)[0] - 1,
#                 ]
#         return np.dot(self.magic_array, bin_indices)
#
#     def getBinIndices(self, linear_index):
#         """Given a linear index (integer between 0 and outdim), returns the bin
#         indices for each of the state dimensions.
#
#         """
#         return linear_index / self.magic_array % self.nbins_across_dims
#
#     def getObservation(self):
#         (theta, thetad, omega, omegad, omegadd,
#                 xf, yf, xb, yb, psi) = self.env.getSensors()
#         state = one_to_n(self.getBin(theta, thetad, omega, omegad, omegadd),
#                 self.outdim)
#
#         return state