import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint

# Define process model (SISO FOPDT)
def process_model(y,t,u,K,tau):
    # arguments
    #  y   = outputs
    #  t   = time
    #  u   = input value
    #  K   = process gain
    #  tau = process time constant

    # calculate derivative
    dydt = (-y + K * u)/tau

    return dydt

class ProcessModel:
    def __init__(self, K, tau, dt):
        self.y0 = 0
        self.K = K
        self.tau = tau
        self.dt = dt
        

    def run(self, u):
        y = odeint(process_model,self.y0,[0, self.dt],args=(u,self.K,self.tau))
        self.y0 = y[-1]
        return y[-1]
        