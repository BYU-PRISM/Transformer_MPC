import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from sklearn import preprocessing

import numpy as np
import time
from scipy.integrate import odeint
from scipy.optimize import minimize


class Mpc_nn:
    def __init__(self, window, nu, ny, P, M, s1, s2, multistep=0, model_one=None, model_multi=None):
        self.y_hat0 = 0
        self.window = window
        self.nu = nu
        self.ny = ny
        self.P = P
        self.M = M
        self.multistep = multistep
        self.model_one = model_one
        self.model_multi = model_multi
        self.s1 = s1
        self.s2 = s2

    def MPCobj_nn(self, u_hat, u_window, y_window, sp):
        # future u values after the control horizon
        u_hat = np.reshape(u_hat, (self.M, self.nu))
        u_hat_P = np.ones((self.P - self.M, self.nu)) * u_hat[-1]
        u_all = np.concatenate((u_window, u_hat, u_hat_P), axis=0)

        y_hat = np.ones((self.P, self.ny)) * y_window[-1]
        y_all = np.concatenate((y_window, y_hat), axis=0)

        SP_hat = np.ones((self.P, self.ny)) * sp

        X = np.concatenate((u_all, y_all), axis=1)
        Y = y_all
        SP_t = SP_hat

        Xs = self.s1.transform(X)
        Ys = self.s2.transform(Y)
        SPs = self.s2.transform(SP_t)

        # Appending the window (past) and Prediction (future) arrays
        Xsq = Xs.copy()
        Ysq = Ys.copy()

        if self.multistep == 0:
            # SPsq = np.reshape(SP_pred, (P,Ys.shape[1]))
            for i in range(self.window, len(Xsq)):
                Xin = Xsq[i - self.window:i].reshape((1, self.window, np.shape(Xsq)[1]))
                # LSTM or Transformer prediction
                Xsq[i][(Xs.shape[1] - Ys.shape[1]):] = self.model_one(Xin)
                # (Xs.shape[1]-Ys.shape[1]) indicates the index of the 
                # first 'system' output variable in the 'LSTM (Transformer)' input array
                Ysq[i] = Xsq[i][(Xs.shape[1] - Ys.shape[1]):]

        else:
            Xin = Xsq.reshape((1, self.window + self.P, np.shape(Xsq)[1]))
            Ysq = self.model_multi(Xin)[0]
            # print(Ysq)

        Ytu = self.s2.inverse_transform(Ysq)
        Xtu = self.s1.inverse_transform(Xsq)

        # print(Ytu)

        u_hat0 = np.append(u_window[-1], u_hat)  # prepare for 'rate of change of MV' in the objective function
        u_hat0 = u_hat0.reshape((-1, self.nu))

        W_CV = np.array([4, 10])
        W_MV = np.array([15, 15])

        pred_nn = {}
        if self.multistep == 0:
            pred_nn["y_hat"] = Ytu[self.window:]
            # pred_nn["u_hat"] = Xtu[self.window:, 0:self.nu]

            Obj = np.sum(((pred_nn["y_hat"] - SP_hat) ** 2).dot(W_CV)) + np.sum(
                ((u_hat0[1:] - u_hat0[0:-1]) ** 2).dot(W_MV))

        else:
            pred_nn["y_hat_multi"] = Ytu
            # pred_nn["u_hat_multi"] = Xtu[self.window:,0]

            Obj = np.sum(((pred_nn["y_hat_multi"] - SP_hat) ** 2).dot(W_CV)) + np.sum(
                ((u_hat0[1:] - u_hat0[0:-1]) ** 2).dot(W_MV))
        print('Obj=', Obj)
        return Obj

    def run(self, uhat, u_window, y_window, sp):
        # MPC calculation

        # u_hat0 = np.ones(self.M) * ui
        start = time.time()
        bnds = np.array([[0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3]])
        # solution = minimize(self.objective,ui,method='SLSQP', args=(yp,sp))
        # solution = minimize(self.MPCobj_nn, uhat, method='SLSQP',bounds=bnds,args=(u_window, y_window, sp),options={'eps': 1e-06, 'ftol': 1e-01})
        solution = minimize(self.MPCobj_nn, uhat, method='SLSQP', bounds=bnds, args=(u_window, y_window, sp),
                            options={'eps': 1e-06,
                                     'maxiter': 100,
                                     'ftol': 1e-03})
        # 

        u = solution.x
        u = np.reshape(u, (4, 2))

        end = time.time()
        elapsed = end - start
        print(elapsed)

        return u

    def RunNN(self, u_hat, u_window, y_window, sp):
        # future u values after the control horizon
        u_hat = np.reshape(u_hat, (self.M, self.nu))
        u_hat_P = np.ones((self.P - self.M, self.nu)) * u_hat[-1]
        u_all = np.concatenate((u_window, u_hat, u_hat_P), axis=0)

        y_hat = np.ones((self.P, self.ny)) * y_window[-1]
        y_all = np.concatenate((y_window, y_hat), axis=0)

        SP_hat = np.ones((self.P, self.ny)) * sp

        X = np.concatenate((u_all, y_all), axis=1)
        Y = y_all
        SP_t = SP_hat

        Xs = self.s1.transform(X)
        Ys = self.s2.transform(Y)
        SPs = self.s2.transform(SP_t)

        # Appending the window (past) and Prediction (future) arrays
        Xsq = Xs.copy()
        Ysq = Ys.copy()

        if self.multistep == 0:
            # SPsq = np.reshape(SP_pred, (P,Ys.shape[1]))
            for i in range(self.window, len(Xsq)):
                Xin = Xsq[i - self.window:i].reshape((1, self.window, np.shape(Xsq)[1]))
                # LSTM or Transformer prediction
                Xsq[i][(Xs.shape[1] - Ys.shape[1]):] = self.model_one(Xin)
                # (Xs.shape[1]-Ys.shape[1]) indicates the index of the 
                # first 'system' output variable in the 'LSTM (Transformer)' input array
                Ysq[i] = Xsq[i][(Xs.shape[1] - Ys.shape[1]):]

        else:
            Xin = Xsq.reshape((1, self.window + self.P, np.shape(Xsq)[1]))
            Ysq = self.model_multi(Xin)[0]

        Ytu = self.s2.inverse_transform(Ysq)
        Xtu = self.s1.inverse_transform(Xsq)

        return Ytu
