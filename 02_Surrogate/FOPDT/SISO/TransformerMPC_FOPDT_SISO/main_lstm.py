from siso_fopdt import *
from mpc_fopdt import Mpc


import numpy as np
import matplotlib.pyplot as plt
import time
# from scipy.integrate import odeint
import 
from scipy.optimize import minimize

# FOPDT Parameters
K=3.0      # gain
tau=5.0    # time constant
ns = 100    # Simulation Length
t = np.linspace(0,ns,ns+1)
delta_t = t[1]-t[0]


# Define horizons
P = 12 # Prediction Horizon
M = 5  # Control Horizon

# Input Sequence
u = np.zeros(ns+1)
# u[5:] = 5

# Setpoint Sequence

sp = np.zeros(ns+1)
sp[10:40] = 2
sp[40:80] = 10
sp[80:] = 3
# Controller setting
maxmove = 1

## Process simulation 
yp = np.zeros(ns+1)


p = ProcessModel(K, tau, delta_t)
m = Mpc(P, M, K, tau, delta_t)

uhat = np.zeros(M)

for i in range(1,ns):
    print(i)
    # run process model
    yp[i+1] = p.run(u[i])

    # run MPC 
    uhat = m.run(uhat, yp[i], sp[i])
    u[i+1] = uhat[0]
    delta = u[i+1] - u[i]
    
    if np.abs(delta) >= maxmove:
        if delta > 0:
            u[i+1] = u[i]+maxmove
        else:
            u[i+1] = u[i]-maxmove

    # else:
    #     u[i+1] = u[i]+delta[0]   


    

plt.step(t, yp)
plt.step(t, u)
plt.step(t, sp)
plt.show()
 


print(yp[7:13], u[7:13], sp[7:13])



