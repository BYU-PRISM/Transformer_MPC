import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
import warnings

warnings.filterwarnings('ignore')

class lstm_pred():
    Ore_amps = []
    Sulfur_tph = []
    O2_scfm = []
    Carbon_in = []
    Sulf_in = []
    CO3_in = []
    O2 = []
    CO2 = []
    SO2 = []
    TCM = []
    FeS2 = []
    CaCO3 = []
    T_1 = []
    T_2 = []
    
#%% Using predicted values to predict next step
def rto(past, Dof, tail, window, sp, v, s1, P, M, ctl, umeas, nCV):
    steps = {}
    for key in past.keys():
        steps[key] = np.concatenate((past[key], Dof[key], tail[key]))
    
    steps = pd.DataFrame(data = steps)

    Xt = steps[['Ore_amps', 'Sulfur_tph', 'O2_scfm', 'Carbon_in','Sulf_in', 'CO3_in',\
                'O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
    # Yt = steps[['O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
    
        
    Xt = np.array(Xt)
        
    Dof = np.array([Xt[window:window+M,0], Xt[window:window+M,1], Xt[window:window+M,2]])
    
    Ore_amps_upper = np.ones(M) * 100 
    Ore_amps_lower = np.ones(M) * 80
    Sulfur_tph_upper = np.ones(M) * 13
    Sulfur_tph_lower = np.ones(M) * 9
    O2_scfm_upper = np.ones(M) * 8000 
    O2_scfm_lower = np.ones(M) * 7000
    
    # lower_bnds = np.concatenate([Sulfur_tph_lower])
    # upper_bnds = np.concatenate([Sulfur_tph_upper])
    
    lower_bnds = np.concatenate([Ore_amps_lower, Sulfur_tph_lower, O2_scfm_lower])
    upper_bnds = np.concatenate([Ore_amps_upper, Sulfur_tph_upper, O2_scfm_upper])
    
    bnds = np.vstack([lower_bnds, upper_bnds]).T

    if ctl == 0:
        Dof = np.reshape(Dof, (1,3*M))[0]
        sol = predictions(Dof, Xt, window, sp, v, s1, P, M, umeas, nCV)
        
    else:
        sol = minimize(predictions, Dof, args=(Xt, window, sp, v, s1, P, M, umeas, nCV), bounds = bnds, method='SLSQP', options={'ftol':1e-3, 'disp':True})
        
        lstm_pred.Ore_amps[0:M] = sol.x[0:M]
        lstm_pred.Sulfur_tph[0:M] = sol.x[M:2*M]
        lstm_pred.O2_scfm[0:M] = sol.x[2*M:3*M]
        
        lstm_pred.Ore_amps[M:] =  lstm_pred.Ore_amps[M-1]
        lstm_pred.Sulfur_tph[M:] = lstm_pred.Sulfur_tph[M-1]
        lstm_pred.O2_scfm[M:] = lstm_pred.O2_scfm[M-1]
      
    print(lstm_pred.O2)
    print(sol)
    return lstm_pred
        

def predictions(Dof, Xt, window, sp, v, s1,  P, M, umeas, nCV):
    
    Xt[window:window+M,0] = Dof[0:M]
    Xt[window:window+M,1] = Dof[M:2*M]
    Xt[window:window+M,2] = Dof[2*M:3*M]
    
    Xt[window+M:window+P,0] = Dof[M-1]
    Xt[window+M:window+P,1] = Dof[2*M-1]
    Xt[window+M:window+P,2] = Dof[3*M-1]
    
    
    Xts = s1.transform(Xt)
    
    Xtp = np.zeros([P,14])
    # Ytp = np.zeros([P,8])
    for i in range(window,P+window):
        Xin = Xts[i-window:i].reshape((1, window, 14))
        Xts[i][6:] = v.predict(Xin)
        Xtp[i-window] = Xts[i]
        # Ytp[i-window] = Xts[i][6:]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        Xtu = s1.inverse_transform(Xtp)
        
    lstm_pred.Ore_amps = Xtu[:,0]
    lstm_pred.Sulfur_tph = Xtu[:,1]
    lstm_pred.O2_scfm = Xtu[:,2]
    lstm_pred.Carbon_in = Xtu[:,3]
    lstm_pred.Sulf_in = Xtu[:,4]
    lstm_pred.CO_in = Xtu[:,5]
    lstm_pred.O2 = Xtu[:,6] + np.ones(P)*umeas["O2"]
    lstm_pred.CO2 = Xtu[:,7] + np.ones(P)*umeas["CO2"]
    lstm_pred.SO2 = Xtu[:,8] + np.ones(P)*umeas["SO2"]
    lstm_pred.TCM = Xtu[:,9] + np.ones(P)*umeas["TCM"]
    lstm_pred.FeS2 = Xtu[:,10] + np.ones(P)*umeas["FeS2"]
    lstm_pred.CaCO3 = Xtu[:,11] + np.ones(P)*umeas["CaCO3"]
    lstm_pred.T_1 = Xtu[:,12] + np.ones(P)*umeas["T_1"]
    lstm_pred.T_2 = Xtu[:,13] + np.ones(P)*umeas["T_2"]
    
    delta_Ore_amps = np.diff(lstm_pred.Ore_amps)
    delta_Sulfur_tph = np.diff(lstm_pred.Sulfur_tph)
    delta_O2_scfm = np.diff(lstm_pred.O2_scfm)
    
    SSD_Ore_amps = np.sum(delta_Ore_amps**2)
    SSD_Sulfur_tph = np.sum(delta_Sulfur_tph**2)
    SSD_O2_scfm = np.sum(delta_O2_scfm**2)
    
    # print(lstm_pred.Ore_amps)
    # obj = -1*lstm_pred.Ore_amps[-1] 
    
    pred_nn = np.array([lstm_pred.O2, lstm_pred.CO2, lstm_pred.SO2 ]).T
    ssd_nn = np.array([delta_Ore_amps, delta_Sulfur_tph, delta_O2_scfm]).T

    W_CV = np.array([1e-2, 1e-2, 1e-2])
    W_MV = np.array([1e-2, 1e-2, 1e-3])

    SSE = np.sum(((sp - pred_nn)**2).dot(W_CV))
    SSD = np.sum(((ssd_nn)**2).dot(W_MV))
    
    obj = SSE + SSD 
    print(obj)
    print(pred_nn, ssd_nn)
    return obj


# def sim(steps, window, nstep, v, s1, s2, P, M):
#     steps = pd.DataFrame(data = steps)
#     Xt = steps[['Ore_amps', 'Sulfur_tph', 'O2_scfm', 'Carbon_in','Sulf_in', 'CO3_in',\
#                 'O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
#     # Yt = steps[['O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
    
#     Xts = s1.transform(Xt)
#     # Yts = s2.transform(Yt)
    
#     Xtp = np.zeros([nstep-window,14])
#     # Ytp = np.zeros([nstep-window,8])
    
#     for i in range(window,nstep):
#         Xin = Xts[i-window:i].reshape((1, window, 14))
#         Xts[i][6:] = v.predict(Xin)
#         Xtp[i-window] = Xts[i]
#         # Ytp[i-window] = Xts[i][6:]
                        
#     Xtu = s1.inverse_transform(Xtp)
#     # Ytu = s2.inverse_transform(Ytp)
            
#     lstm_pred.Ore_amps = Xtu[:,0]
#     lstm_pred.Sulfur_tph = Xtu[:,1]
#     lstm_pred.O2_scfm = Xtu[:,2]
#     lstm_pred.Carbon_in = Xtu[:,3]
#     lstm_pred.Sulf_in = Xtu[:,4]
#     lstm_pred.CO_in = Xtu[:,5]
#     lstm_pred.O2 = Xtu[:,6]
#     lstm_pred.CO2 = Xtu[:,7]
#     lstm_pred.SO2 = Xtu[:,8]
#     lstm_pred.TCM = Xtu[:,9]
#     lstm_pred.FeS2 = Xtu[:,10]
#     lstm_pred.CaCO3 = Xtu[:,11]
#     lstm_pred.T_1 = Xtu[:,12]
#     lstm_pred.T_2 = Xtu[:,13]
    
    
#     return lstm_pred 

