import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize



# Define MPC model (SISO FOPDT, same as process model in this case)
def mpc_model(y,t,u,K,tau):
    # arguments
    #  y   = outputs
    #  t   = time
    #  u   = input value
    #  K   = process gain
    #  tau = process time constant

    # calculate derivative
    dydt = (-y + K * u)/tau

    return dydt


class Mpc:
    def __init__(self, P, M, K, tau, dt):
        self.y_hat0 = 0
        self.P = P
        self.M = M
        self.K = K
        self.tau = tau
        self.dt = dt
        # self.yp = yp
        # self.sp = sp


    # Define Objective function      
    def objective(self, u_hat0, yp, sp):
        
        u_hat = np.ones(self.P) * u_hat0[-1]
        u_hat[0:self.M] = u_hat0
        yp_hat = np.zeros(self.P)
        sp_hat = np.zeros(self.P)
        se = np.zeros(self.P)
        self.y_hat0 = yp
        # Prediction
        for k in range(0, self.P):
                
            y_hat = odeint(mpc_model,self.y_hat0,[0,self.dt],args=(u_hat[k],self.K,self.tau))
            self.y_hat0 = y_hat[-1]
            yp_hat[k] = y_hat[0]

            # Squared Error calculation
            sp_hat[k] = sp
            delta_u_hat = np.zeros(self.P)        

            delta_u_hat[k] = u_hat[k]-u_hat[k-1] 
            se[k] = (sp_hat[k]-yp_hat[k])**2 + 50 * (delta_u_hat[k])**2
            # print('k=', k)

        # Sum of Squared Error calculation       
        obj = np.sum(se)
        # print(obj)
        return obj 


    def run(self, ui, yp, sp):
    # MPC calculation

        # u_hat0 = np.ones(self.M) * ui
        start = time.time()

        solution = minimize(self.objective,ui,method='SLSQP', args=(yp,sp))
        u = solution.x  

        end = time.time()
        elapsed = end - start

        

          


        return u

