from siso_fopdt import *
from mpc_fopdt import Mpc


import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize

# FOPDT Parameters
K=3.0      # gain
tau=5.0    # time constant
ns = 20    # Simulation Length
t = np.linspace(0,ns,ns+1)
delta_t = t[1]-t[0]

# Define horizons
P = 30 # Prediction Horizon
M = 10  # Control Horizon

# Input Sequence
u = np.zeros(ns+1)
# u[5:] = 5

# Setpoint Sequence
sp = np.zeros(ns+1+2*P)
sp[10:40] = 5
sp[40:80] = 10
sp[80:] = 3
# Controller setting
maxmove = 1

## Process simulation 
yp = np.zeros(ns+1)


p = ProcessModel(K, tau, delta_t)
m = Mpc(P, M, K, tau, delta_t)

uhat = np.zeros(M)

for i in range(1,ns+1):
    print(i)
    # run process model
    yp[i] = p.run(u[i-1])

    # run MPC 
    uhat = m.run(uhat, yp[i], sp[i])
    u[i] = uhat[0]
    delta = np.diff(uhat)
    
    if np.abs(delta[0]) >= maxmove:
        if delta[0] > 0:
            u[i] = u[i-1]+maxmove
        else:
            u[i] = u[i-1]-maxmove

    # else:
    #     u[i+1] = u[i]+delta[0]   
    
    
    

plt.plot(yp)
plt.plot(u)
plt.plot(sp)
plt.show()
 
