from hashlib import shake_128
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import odeint
from scipy.optimize import minimize


class Mpc:
    def __init__(self, window, P, M, dt, s1, s2,  model=None):
        self.y_hat0 = 0
        self.P = P
        self.M = M
        self.dt = dt
        self.model = model
        self.s1 = s1 
        self.s2 = s2
        self.window = window


        # self.yp = yp
        # self.sp = sp

    def MPCobj_lstm(u_hat, y_hat, SP_hat, u, y, SP, window, P, M, multistep):
        # future u values after the control horizon
        u_hat_P = np.ones(P-M) * u_hat[-1]
        u_all = np.concatenate((u, u_hat, u_hat_P),axis=None)
        y_all = np.append(y, y_hat)

        # X = pd.DataFrame({'u': u_all, 'y':y_all})
        # Y = pd.DataFrame({'y': y_all})

        X = np.transpose([u_all,y_all]) 
        Y = np.transpose([y_all])
        SP_trans = np.transpose([SP_hat])

        Xs = s1.transform(X)
        Ys = s2.transform(Y)
        SPs = s2.transform(SP_trans)

        # Appending the window (past) and Prediction (future) arrays
        Xsq = Xs.copy()
        Ysq = Ys.copy()

        if multistep == 0:
            # SPsq = np.reshape(SP_pred, (P,Ys.shape[1]))
            for i in range(window,len(Xsq)):
                Xin = Xsq[i-window:i].reshape((1, window, np.shape(Xsq)[1]))
                # LSTM prediction
                Xsq[i][(Xs.shape[1] - Ys.shape[1]):] = model_lstm(Xin) 
                # (Xs.shape[1]-Ys.shape[1]) indicates the index of the 
                # first 'system' output variable in the 'LSTM' input array
                Ysq[i] = Xsq[i][(Xs.shape[1] - Ys.shape[1]):]
        
        else:
            Xin = Xsq.reshape((1, window+P, np.shape(Xsq)[1]))
            Ysq = model_lstm_multi(Xin)


        Ytu = s2.inverse_transform(Ysq)
        Xtu = s1.inverse_transform(Xsq)

        u_hat0 = np.append(u[-1], u_hat)

        if multistep == 0:
            pred_lstm["y_hat"] = np.reshape(Ytu[window:], (1,P))[0]
            pred_lstm["u_hat"] = np.reshape(Xtu[window:,0], (1,P))[0]

            Obj = 10*np.sum((pred_lstm["y_hat"] - SP_hat)**2) + np.sum(((u_hat0[1:] - u_hat0[0:-1])**2))

        else:
            pred_lstm["y_hat_multi"] = Ytu[0]
            pred_lstm["u_hat_multi"] = Xtu[window:,0]
            
            Obj = 10*np.sum((pred_lstm["y_hat_multi"] - SP_hat)**2) + np.sum(((u_hat0[1:] - u_hat0[0:-1])**2))

        return Obj
    
    


    def run(self, ui, yp, sp):
    # MPC calculation

        # u_hat0 = np.ones(self.M) * ui
        start = time.time()

        solution = minimize(self.objective,ui,method='SLSQP', args=(yp,sp))
        u = solution.x  

        end = time.time()
        elapsed = end - start

        

          


        return u

