import numpy as np
import time
from scipy.integrate import odeint
from scipy.optimize import minimize


class Mpc_nn:
    def __init__(self, window, P, M, s1, s2, multistep=False, model_one=None, model_multi=None):
        self.y_hat0 = 0
        self.window = window
        self.P = P
        self.M = M
        self.multistep = multistep
        self.model_one = model_one
        self.model_multi = model_multi
        self.s1 = s1 
        self.s2 = s2
        
    def MPCobj_nn(self, u_hat, u_window, y_window, sp):
        # future u values after the control horizon
        u_hat_P = np.ones(self.P-self.M) * u_hat[-1]
        u_all = np.concatenate((u_window, u_hat, u_hat_P),axis=None)

        y_hat = np.ones(self.P) * y_window[-1]
        y_all = np.append(y_window, y_hat)

        SP_hat = np.ones(self.P) * sp

        X = np.transpose([u_all,y_all]) 
        Y = np.transpose([y_all])
        SP_t = np.transpose([SP_hat])

        Xs = self.s1.transform(X)
        Ys = self.s2.transform(Y)
        SPs = self.s2.transform(SP_t)

        # Appending the window (past) and Prediction (future) arrays
        Xsq = Xs.copy()
        Ysq = Ys.copy()

        if self.multistep == 0:
            # SPsq = np.reshape(SP_pred, (P,Ys.shape[1]))
            for i in range(self.window,len(Xsq)):
                Xin = Xsq[i-self.window:i].reshape((1, self.window, np.shape(Xsq)[1]))
                # LSTM or Transformer prediction
                Xsq[i][(Xs.shape[1] - Ys.shape[1]):] = self.model_one(Xin) 
                # (Xs.shape[1]-Ys.shape[1]) indicates the index of the 
                # first 'system' output variable in the 'LSTM (Transformer)' input array
                Ysq[i] = Xsq[i][(Xs.shape[1] - Ys.shape[1]):]
        
        else:
            Xin = Xsq.reshape((1, self.window+self.P, np.shape(Xsq)[1]))
            Ysq = self.model_multi(Xin)


        Ytu = self.s2.inverse_transform(Ysq)
        Xtu = self.s1.inverse_transform(Xsq)

        u_hat0 = np.append(u_window[-1], u_hat)


        pred_nn = {}
        if self.multistep == 0:
            pred_nn["y_hat"] = np.reshape(Ytu[self.window:], (1,self.P))[0]
            pred_nn["u_hat"] = np.reshape(Xtu[self.window:,0], (1,self.P))[0]

            Obj = 10*np.sum((pred_nn["y_hat"] - SP_hat)**2) + np.sum(((u_hat0[1:] - u_hat0[0:-1])**2))

        else:
            pred_nn["y_hat_multi"] = Ytu[0]
            pred_nn["u_hat_multi"] = Xtu[self.window:,0]
            
            Obj = 10*np.sum((pred_nn["y_hat_multi"] - SP_hat)**2) + np.sum(((u_hat0[1:] - u_hat0[0:-1])**2))

        return Obj
    
    


    def run(self,uhat, u_window, y_window, sp):
    # MPC calculation

        # u_hat0 = np.ones(self.M) * ui
        start = time.time()

        # solution = minimize(self.objective,ui,method='SLSQP', args=(yp,sp))
        solution = minimize(self.MPCobj_nn, uhat, method='SLSQP',args=(u_window, y_window, sp),options={'eps': 1e-06, 'ftol': 1e-01})
        u = solution.x  

        end = time.time()
        elapsed = end - start
        print(elapsed)

        return u

