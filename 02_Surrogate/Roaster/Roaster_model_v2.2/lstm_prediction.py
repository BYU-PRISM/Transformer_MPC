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
    delta_MV1 = []
    
#%% Using predicted values to predict next step
def rto(past, Dof, tail, window, sp, v, s1, s2, P, M, ctl, umeas, nCV, nMV, multistep=0):
    steps = {}
    for key in past.keys():
        steps[key] = np.concatenate((past[key], Dof[key], tail[key]))
    
    steps = pd.DataFrame(data = steps)

    Xt = steps[['Ore_amps', 'Sulfur_tph', 'O2_scfm', 'Carbon_in','Sulf_in', 'CO3_in',\
                'O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
    Yt = steps[['O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]
    
        
    Xt = np.array(Xt)
    Yt = np.array(Yt)
        
    Dof1 = np.array(Xt[window:window+M, 0:nMV]).T
    # Dof1 = np.array(Xt[window:window+M, 1]).T
    
    Ore_amps_upper = np.ones(M) * 100
    Ore_amps_lower = np.ones(M) * 80
    Sulfur_tph_upper = np.ones(M) * 13
    Sulfur_tph_lower = np.ones(M) * 9
    # O2_scfm_upper = np.ones(M) * 7401 
    # O2_scfm_lower = np.ones(M) * 7400
    
    # lower_bnds = np.concatenate([Sulfur_tph_lower])
    # upper_bnds = np.concatenate([Sulfur_tph_upper])
    
    # lower_bnds = np.concatenate([Ore_amps_lower,Sulfur_tph_lower,O2_scfm_lower])
    # upper_bnds = np.concatenate([Ore_amps_upper,Sulfur_tph_upper,O2_scfm_upper])

    lower_bnds = np.concatenate([Ore_amps_lower,Sulfur_tph_lower])
    upper_bnds = np.concatenate([Ore_amps_upper,Sulfur_tph_upper])

    
    bnds = np.vstack([lower_bnds, upper_bnds]).T
    Dof1 = np.reshape(Dof1, (1,-1))[0]
    if ctl == 0:
        sol = predictions(Dof1, Xt, window, sp, v, s1, s2, P, M, umeas, nCV, nMV, multistep)
        
    else:
        sol = minimize(predictions, Dof1, bounds=bnds, args=(Xt, window, sp, v, s1, s2, P, M, umeas, nCV, nMV, multistep), method='SLSQP', options={'eps': 1e-03,'ftol':1e-1, 'disp':True})
        
        lstm_pred.Ore_amps[window:window+M] = sol.x[0:M]
        lstm_pred.Sulfur_tph[window:window+M] = sol.x[M:2*M]
        # lstm_pred.O2_scfm[window:window+M] = sol.x[2*M:3*M]
        
        lstm_pred.Ore_amps[window+M:] =  lstm_pred.Ore_amps[window+M-1]
        lstm_pred.Sulfur_tph[window+M:] = lstm_pred.Sulfur_tph[window+M-1]
        # lstm_pred.O2_scfm[window+M:] = lstm_pred.O2_scfm[window+M-1]
      
    # print(lstm_pred.Sulfur_tph)
    print(sol)
    return lstm_pred
        

def predictions(Dof, Xt, window, sp, v, s1, s2, P, M, umeas, nCV, nMV, multistep):
    
    Xt[window:window+M,0] = Dof[0:M]
    Xt[window:window+M,1] = Dof[M:2*M]
    # Xt[window:window+M,2] = Dof[2*M:3*M]
    
    Xt[window+M:window+P,0] = Dof[M-1]
    Xt[window+M:window+P,1] = Dof[2*M-1]
    # Xt[window+M:window+P,2] = Dof[3*M-1]

    # Xt[window:window+M,1] = Dof
    # Xt[window+M:window+P,1] = Dof[-1]
    
    # Xts = s1.transform(Xt)
    Xsq = (Xt - s1.data_min_)/(s1.data_max_-s1.data_min_)*2-1

    Xtp = np.zeros([P,14])
    Ytp = np.zeros([P,8])
    Yts = np.zeros([P,8])

    Xts = Xsq.copy()
    # Yts = Ysq.copy()
    # Ytp = np.zeros([P,8])
    if multistep == 0:
        for i in range(window,P+window):
            Xin = Xts[i-window:i].reshape((1, window, 14))
            Xts[i][6:] = v.predict(Xin)
            Xtp[i-window] = Xts[i]
            # Ytp[i-window] = Xts[i][6:]

    else:
        Xin = Xts.reshape((1, window + P, np.shape(Xts)[1]))
        Xts[window:, 6:] = v(Xin)[0]
        # Yts = Xts[window:, 6:] 
        

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # Xtu = s1.inverse_transform(Xtp)

        Xtu = (Xts +1)/2 * (s1.data_max_ - s1.data_min_)+ s1.data_min_ 
        # Ytu = (Yts +1)/2 * (s2.data_max_ - s2.data_min_)+ s2.data_min_ 
        
    lstm_pred.Ore_amps = Xtu[:,0]
    lstm_pred.Sulfur_tph = Xtu[:,1]
    lstm_pred.O2_scfm = Xtu[:,2]
    lstm_pred.Carbon_in = Xtu[:,3]
    lstm_pred.Sulf_in = Xtu[:,4]
    lstm_pred.CO_in = Xtu[:,5]
    lstm_pred.O2 = Xtu[:,6]
    lstm_pred.CO2 = Xtu[:,7]
    lstm_pred.SO2 = Xtu[:,8]
    lstm_pred.TCM = Xtu[:,9]
    lstm_pred.FeS2 = Xtu[:,10]
    lstm_pred.CaCO3 = Xtu[:,11]
    lstm_pred.T_1 = Xtu[:,12]
    lstm_pred.T_2 = Xtu[:,13]

    # lstm_pred.O2 += umeas["O2"]
    # lstm_pred.CO2 += umeas["CO2"]
    # lstm_pred.SO2 += umeas["SO2"]
    # lstm_pred.TCM += umeas["TCM"]
    # lstm_pred.FeS2 += umeas["FeS2"]
    # lstm_pred.CaCO3 += umeas["CaCO3"]
    # lstm_pred.T_1 += umeas["T_1"]
    # lstm_pred.T_2 += umeas["T_2"]

    
    
    # MV = Xtu[window:window+M,0:nMV]
    # MV = Xtu[window:window+M,1]
    
    # delta_MV = np.diff(MV, axis=0)

    delta_Ore_amps = np.diff(lstm_pred.Ore_amps[window:window+M])
    delta_Sulfur_tph = np.diff(lstm_pred.Sulfur_tph[window:window+M])
    # delta_O2_scfm = np.diff(lstm_pred.O2_scfm[window:window+M])
    
    # SSD_Ore_amps = np.sum(delta_Ore_amps**2)
    # SSD_Sulfur_tph = np.sum(delta_Sulfur_tph**2)
    # SSD_O2_scfm = np.sum(delta_O2_scfm**2)
    
    # print(lstm_pred.Ore_amps)
    # obj = -1*lstm_pred.Ore_amps[-1] 
    
    pred_nn = np.array([lstm_pred.T_1[window:], lstm_pred.T_2[window:]]).T
    # pred_nn = lstm_pred.T_1[window:] #+ umeas["T_1"]
    # ssd_nn = np.array([delta_Ore_amps, delta_Sulfur_tph, delta_O2_scfm]).T
    ssd_nn = np.array([delta_Ore_amps, delta_Sulfur_tph]).T

    W_CV = np.array([1e3, 1e2])
    # W_MV = np.array([1e1, 1e0, 1e3])
    W_MV = np.array([1e3, 1e0])
    # W_CV = 1e3
    # W_MV = 1e0

    SSE = np.sum(((sp - pred_nn)**2).dot(W_CV))
    SSD = np.sum(((ssd_nn)**2).dot(W_MV))
    # SSD = np.sum(((delta_MV)**2).dot(W_MV))
    # SSD = np.sum(W_MV*(delta_Dof**2))
    
    obj = SSE + SSD 
    print(obj)
    # print(pred_nn, ssd_nn)
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

