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

        yp_hat[0] = self.y_hat0

        # dt0 = (self.P*self.dt)**(1/self.P)
        # t = np.array([dt0**(i+1) for i in range(self.P)])
        # dt = [[t[i], t[i-1]] for i in range(len(t))]

        # Prediction
        for k in range(0, self.P-1):
                
            y_hat = odeint(mpc_model,self.y_hat0,[0,self.dt],args=(u_hat[k],self.K,self.tau))
            self.y_hat0 = y_hat[-1]
            yp_hat[k+1] = y_hat[-1]

            # Squared Error calculation
            sp_hat[k] = sp
            delta_u_hat = np.zeros(self.P)        

            delta_u_hat[k] = u_hat[k]-u_hat[k-1] 
            se[k] = (sp_hat[k]-yp_hat[k])**2 + 10 * (delta_u_hat[k])**2
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

