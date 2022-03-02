import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize



# Define process model
def process_model(y,t,u,K,tau):
    # arguments
    #  y   = outputs
    #  t   = time
    #  u   = input value
    #  K   = process gain
    #  tau = process time constant

    # calculate derivative
    dydt = (-y + K * u)/(tau)

    return dydt 

# Define Objective function      
def objective(u_hat):
    # Prediction
    for k in range(1,2*P+1):
        if k==1:
            y_hat0 = yp[i-P]

        if k<=P:
            if i-P+k<0:
                u_hat[k] = 0

            else: 
                u_hat[k] = u[i-P+k]

        elif k>P+M:
            u_hat[k] = u_hat[P+M]

        ts_hat = [delta_t_hat*(k-1),delta_t_hat*(k)]        
        y_hat = odeint(process_model,y_hat0,ts_hat,args=(u_hat[k],K,tau))
        y_hat0 = y_hat[-1]
        yp_hat[k] = y_hat[0]

        # Squared Error calculation
        sp_hat[k] = sp[i]
        delta_u_hat = np.zeros(2*P+1)        

        if k>P:
            delta_u_hat[k] = u_hat[k]-u_hat[k-1] 
            se[k] = (sp_hat[k]-yp_hat[k])**2 + 20 * (delta_u_hat[k])**2

    # Sum of Squared Error calculation       
    obj = np.sum(se[P+1:])
    return obj 

# FOPDT Parameters
K=3.0      # gain
tau=5.0    # time constant
ns = 20    # Simulation Length
t = np.linspace(0,ns,ns+1)
delta_t = t[1]-t[0]

# Define horizons
P = 10 # Prediction Horizon
M = 10  # Control Horizon

# Input Sequence
u = np.zeros(ns+1)

# Setpoint Sequence
sp = np.zeros(ns+1+2*P)
sp[10:40] = 5
sp[40:80] = 10
sp[80:] = 3
# Controller setting
maxmove = 1

## Process simulation 
yp = np.zeros(ns+1)



for i in range(1,ns+1):
    print(i)
    if i==1:
        y0 = 0
    ts = [delta_t*(i-1),delta_t*i]
    y = odeint(process_model,y0,ts,args=(u[i],K,tau))
    y0 = y[-1]
    yp[i] = y[0]

    # Declare the variables in fuctions
    t_hat = np.linspace(i-P,i+P,2*P+1)
    delta_t_hat = t_hat[1]-t_hat[0]
    se = np.zeros(2*P+1)
    yp_hat = np.zeros(2*P+1)
    u_hat0 = np.zeros(2*P+1) 
    sp_hat = np.zeros(2*P+1)
    obj = 0.0

    # initial guesses
    for k in range(1,2*P+1):

        if k<=P:
            if i-P+k<0:
                u_hat0[k] = 0

            else: 
                u_hat0[k] = u[i-P+k]

        elif k>P:
            u_hat0[k] = u[i]

    # show initial objective
    print('Initial SSE Objective: ' + str(objective(u_hat0))) 

    # MPC calculation
    start = time.time()

    solution = minimize(objective,u_hat0,method='SLSQP')
    u_hat = solution.x  

    end = time.time()
    elapsed = end - start

    print('Final SSE Objective: ' + str(objective(u_hat)))
    print('Elapsed time: ' + str(elapsed) )

    delta = np.diff(u_hat)

    if i<ns:    
        if np.abs(delta[P]) >= maxmove:
            if delta[P] > 0:
                u[i+1] = u[i]+maxmove
            else:
                u[i+1] = u[i]-maxmove

        else:
            u[i+1] = u[i]+delta[P]

plt.plot(yp)
plt.plot(u)
plt.plot(sp)
plt.show()