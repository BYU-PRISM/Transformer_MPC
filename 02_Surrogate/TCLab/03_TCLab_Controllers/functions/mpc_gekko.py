from gekko import GEKKO
from numpy import array, linspace


class MPCModel(GEKKO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dt = 1
        self.time = linspace(0, 15-1, 15*self.dt)

        # Process Gain
        k11 = self.FV(2)
        k12 = self.FV(0)
        k21 = self.FV(0)
        k22 = self.FV(1)

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
        u1 = self.MV(0, lb=0, ub=1)
        u2 = self.MV(0, lb=0, ub=1)

        # FOPDT Equation
        self.Equation(x11.dt() + x11 / tau11 == k11 / tau11 * u1)
        self.Equation(x12.dt() + x12 / tau12 == k12 / tau12 * u2)
        self.Equation(y1 == x11 + x12)

        self.Equation(x21.dt() + x21 / tau21 == k21 / tau21 * u1)
        self.Equation(x22.dt() + x22 / tau22 == k22 / tau22 * u2)
        self.Equation(y2 == x21 + x22)

        self.K11 = k11
        self.K12 = k12
        self.K21 = k21
        self.K22 = k22

        self.tau11 = tau11
        self.tau12 = tau12
        self.tau21 = tau21
        self.tau22 = tau22

        self.u1 = u1
        self.u2 = u2

        self.y1 = y1
        self.y2 = y2

        # STATUS = 0, optimizer doesn't adjust value
        # STATUS = 1, optimizer can adjust
        self.K11.STATUS = 0
        self.K12.STATUS = 0
        self.K21.STATUS = 0
        self.K22.STATUS = 0

        self.tau11.STATUS = 0
        self.tau12.STATUS = 0
        self.tau21.STATUS = 0
        self.tau22.STATUS = 0

        self.u1.STATUS = 1
        self.u2.STATUS = 1

        self.y1.STATUS = 1
        self.y2.STATUS = 1

        # FSTATUS = 0, no measurement
        # FSTATUS = 1, measurement used to update model
        self.K11.FSTATUS = 0
        self.K12.FSTATUS = 0
        self.K21.FSTATUS = 0
        self.K22.FSTATUS = 0

        self.tau11.FSTATUS = 0
        self.tau12.FSTATUS = 0
        self.tau21.FSTATUS = 0
        self.tau22.FSTATUS = 0

        self.u1.FSTATUS = 0
        self.u2.FSTATUS = 0

        self.y1.FSTATUS = 1
        self.y2.FSTATUS = 1

        self.u1.DMAX = 1
        self.u2.DMAX = 1

        self.u1.DCOST = 10
        self.u2.DCOST = 10

        self.y1.TR_INIT = 0
        self.y2.TR_INIT = 0

        self.options.CV_TYPE = 2
        self.options.IMODE = 6

        self.options.NODES = 3
