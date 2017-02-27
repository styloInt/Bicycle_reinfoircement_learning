from __future__ import print_function

"""An attempt to implement Randlov and Alstrom (1998). They successfully
use reinforcement learning to balance a bicycle, and to control it to drive
to a specified goal location. Their work has been used since then by a few
researchers as a benchmark problem.

We only implement the balance task. This implementation differs at least
slightly, since Randlov and Alstrom did not mention anything about how they
annealed/decayed their learning rate, etc. As a result of differences, the
results do not match those obtained by Randlov and Alstrom.

"""

__author__ = 'Chris Dembia, Bruce Cam, Johnny Israeli'

import numpy as np
from scipy import asarray
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign, clip, argwhere
from matplotlib import pyplot as plt

import pybrain.rl.environments
from pybrain.rl.environments.environment import Environment
from pybrain.rl.learners.valuebased.linearfa import SARSALambda_LinFA
from pybrain.rl.learners.valuebased import NFQ, ActionValueNetwork
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.experiments import EpisodicExperiment
from pybrain.utilities import one_to_n

class BicycleEnvironment(Environment):
    """Randlov and Alstrom's bicycle model. This code matches nearly exactly
    some c code we found online for simulating Randlov and Alstrom's
    bicycle. The bicycle travels at a fixed speed.

    """

    # For superclass.
    indim = 2
    outdim = 10

    # Environment parameters.
    time_step = 0.01

    # Goal position and radius
    # Lagouakis (2002) uses angle to goal, not heading, as a state
    max_distance = 1000.

    # Acceleration on Earth's surface due to gravity (m/s^2):
    g = 9.82

    # See the paper for a description of these quantities:
    # Distances (in meters):
    c = 0.66
    dCM = 0.30
    h = 0.94
    L = 1.11
    r = 0.34
    # Masses (in kilograms):
    Mc = 15.0
    Md = 1.7
    Mp = 60.0
    # Velocity of a bicycle (in meters per second), equal to 10 km/h:
    v = 10.0 * 1000.0 / 3600.0

    # Derived constants.
    M = Mc + Mp  # See Randlov's code.
    Idc = Md * r ** 2
    Idv = 1.5 * Md * r ** 2
    Idl = 0.5 * Md * r ** 2
    Itot = 13.0 / 3.0 * Mc * h ** 2 + Mp * (h + dCM) ** 2
    sigmad = v / r
    t=0

    def __init__(self):
        Environment.__init__(self)
        self.reset()
        self.actions = [0.0, 0.0]
        self._save_wheel_contact_trajectories = False

    def performAction(self, actions):
        self.actions = actions
        self.step()

    def _performAction(self, action):
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
        self.performAction([T, d])

    def saveWheelContactTrajectories(self, opt):
        self._save_wheel_contact_trajectories = opt

    def step(self):
        # Unpack the state and actions.
        # -----------------------------
        # Want to ignore the previous value of omegadd; it could only cause a
        # bug if we assign to it.

        (theta, thetad, omega, omegad, _,
         xf, yf, xb, yb, psi) = self.sensors
        (T, d) = self.actions

        # For recordkeeping.
        # ------------------
        if self._save_wheel_contact_trajectories:
            self.xfhist.append(xf)
            self.yfhist.append(yf)
            self.xbhist.append(xb)
            self.ybhist.append(yb)

        # Intermediate time-dependent quantities.
        # ---------------------------------------
        # Avoid divide-by-zero, just as Randlov did.
        if theta == 0:
            rf = 1e8
            rb = 1e8
            rCM = 1e8
        else:
            rf = self.L / np.abs(sin(theta))
            rb = self.L / np.abs(tan(theta))
            rCM = sqrt((self.L - self.c) ** 2 + self.L ** 2 / tan(theta) ** 2)

        phi = omega + np.arctan(d / self.h)

        # Equations of motion.
        # --------------------
        # Second derivative of angular acceleration:
        omegadd = 1 / self.Itot * (self.M * self.h * self.g * sin(phi)
                                   - cos(phi) * (self.Idc * self.sigmad * thetad
                                                 + sign(theta) * self.v ** 2 * (
                                                     self.Md * self.r * (1.0 / rf + 1.0 / rb)
                                                     + self.M * self.h / rCM)))
        thetadd = (T - self.Idv * self.sigmad * omegad) / self.Idl

        # Integrate equations of motion using Euler's method.
        # ---------------------------------------------------
        # yt+1 = yt + yd * dt.
        # Must update omega based on PREVIOUS value of omegad.
        omegad += omegadd * self.time_step
        omega += omegad * self.time_step
        thetad += thetadd * self.time_step
        theta += thetad * self.time_step

        # Handlebars can't be turned more than 80 degrees.
        theta = np.clip(theta, -1.3963, 1.3963)

        # Wheel ('tyre') contact positions.
        # ---------------------------------

        # Front wheel contact position.
        front_temp = self.v * self.time_step / (2 * rf)
        # See Randlov's code.
        if front_temp > 1:
            front_temp = sign(psi + theta) * 0.5 * np.pi
        else:
            front_temp = sign(psi + theta) * arcsin(front_temp)
        xf += self.v * self.time_step * -sin(psi + theta + front_temp)
        yf += self.v * self.time_step * cos(psi + theta + front_temp)

        # Rear wheel.
        back_temp = self.v * self.time_step / (2 * rb)
        # See Randlov's code.
        if back_temp > 1:
            back_temp = np.sign(psi) * 0.5 * np.pi
        else:
            back_temp = np.sign(psi) * np.arcsin(back_temp)
        xb += self.v * self.time_step * -sin(psi + back_temp)
        yb += self.v * self.time_step * cos(psi + back_temp)

        # Preventing numerical drift.
        # ---------------------------
        # Copying what Randlov did.
        current_wheelbase = sqrt((xf - xb) ** 2 + (yf - yb) ** 2)
        if np.abs(current_wheelbase - self.L) > 0.01:
            relative_error = self.L / current_wheelbase - 1.0
            xb += (xb - xf) * relative_error
            yb += (yb - yf) * relative_error

        # Update heading, psi.
        # --------------------
        delta_y = yf - yb
        if (xf == xb) and delta_y < 0.0:
            psi = np.pi
        else:
            if delta_y > 0.0:
                psi = arctan((xb - xf) / delta_y)
            else:
                psi = sign(xb - xf) * 0.5 * np.pi - arctan(delta_y / (xb - xf))

        self.sensors = np.array([theta, thetad, omega, omegad, omegadd,
                                 xf, yf, xb, yb, psi])

    def reset(self):
        theta = 0
        thetad = 0
        omega = 0
        omegad = 0
        omegadd = 0
        xf = 0
        yf = self.L
        xb = 0
        yb = 0
        self.t = 0
        psi = np.arctan((xb - xf) / (yf - yb))
        self.sensors = np.array([theta, thetad, omega, omegad, omegadd,
                                 xf, yf, xb, yb, psi])

        self.xfhist = [xf]
        self.yfhist = [yf]
        self.xbhist = [xb]
        self.ybhist = [yb]

    def getSteer(self):
        return self.sensors[0]

    def getPsi(self):
        return self.sensors[9]

    def getTilt(self):
        return self.sensors[2]

    def get_xfhist(self):
        return self.xfhist

    def get_yfhist(self):
        return self.yfhist

    def get_posf(self):
        return (self.xfhist[-1], self.yfhist[-1])

    def get_posb(self):
        return (self.xbhist[-1], self.ybhist[-1])

    def get_xbhist(self):
        return self.xbhist

    def get_ybhist(self):
        return self.ybhist

    def getSensors(self):
        return self.sensors


