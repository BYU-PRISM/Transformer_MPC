import numpy as np
import matplotlib.pyplot as plt
import time
from gekko import GEKKO


class ProcessModel(GEKKO):
    def __init__(self, dt=1, remote=False):
        super().__init__(remote=remote)

        self.dt = dt
        self.time = np.array([0, self.dt])

        # Process Gain
        K11 = self.FV(2)
        K12 = self.FV(0)
        K21 = self.FV(0)
        K22 = self.FV(1)

        # Time Constant
        tau11 = self.FV(5)
        tau12 = self.FV(5)
        tau21 = self.FV(5)
        tau22 = self.FV(5)

        # # Input Scenario for open-loop
        # u1_input = np.zeros(tf)
        # u1_input[5:] = 1
        # u2_input = np.zeros(tf)
        # u2_input[15:] = -1

        # Setpoint Scenario for closed-loop
        # SP1 = np.zeros(tf)
        # SP2 = np.zeros(tf)

        # SP1[0:] = 0.5
        # SP2[0:] = 0.2
        # SP1[20:] = 0.3
        # SP2[40:] = 0.8

        # Gekko variables for Input Output
        x11 = self.SV(0)
        x12 = self.SV(0)
        x21 = self.SV(0)
        x22 = self.SV(0)
        y1 = self.CV(0)
        y2 = self.CV(0)
        u1 = self.MV(0, lb=0, ub=5)
        u2 = self.MV(0, lb=0, ub=5)

        # FOPDT Equation
        self.Equation(x11.dt() + x11 / tau11 == K11 / tau11 * u1)
        self.Equation(x12.dt() + x12 / tau12 == K12 / tau12 * u2)
        self.Equation(y1 == x11 + x12)

        self.Equation(x21.dt() + x21 / tau21 == K21 / tau21 * u1)
        self.Equation(x22.dt() + x22 / tau22 == K22 / tau22 * u2)
        self.Equation(y2 == x21 + x22)

        self.u1 = u1
        self.u2 = u2

        self.y1 = y1
        self.y2 = y2

        # self.options.CV_TYPE = 2
        self.options.IMODE = 4

        # u1_store = np.ones(tf)*u1.VALUE
        # u2_store = np.ones(tf)*u2.VALUE
        # y1_store = np.ones(tf)*y1.VALUE
        # y2_store = np.ones(tf)*y2.VALUE

        # for i in range(tf):
        #     y1.SP = SP1[i]
        #     y2.SP = SP2[i]

        # self.solve(disp=True)

        # u1_store[i] = u1.NEWVAL
        # u2_store[i] = u2.NEWVAL
        # y1_store[i] = y1.value[1]
        # y2_store[i] = y2.value[1]

        # plt.figure(0)
        # plt.subplot(2,1,1)
        # plt.plot(SP1, drawstyle='steps')
        # plt.plot(SP2, drawstyle='steps')
        # plt.plot(y1_store)
        # plt.plot(y2_store)
        # plt.legend(['SP1', 'SP2', 'y1', 'y2'])

        # plt.subplot(2,1,2)
        # plt.plot(u1_store, drawstyle='steps')
        # plt.plot(u2_store, drawstyle='steps')
        # plt.legend(['u1', 'u2'])
        # plt.show()

    def run(self, u):
        self.u1.value = u[0]
        self.u2.value = u[1]

        self.solve(disp=False)
        # self.time = self.time + self.dt

        return np.array([self.y1.value[-1], self.y2.value[-1]])
