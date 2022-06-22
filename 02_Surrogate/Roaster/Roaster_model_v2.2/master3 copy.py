import Gekko_Roaster_model
import lstm_prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import time

# For LSTM model
from keras.models import load_model

# Make an MP4 animation?
make_mp4 = False
if make_mp4:
    import imageio  # required to make animation
    import os
    try:
        os.mkdir('./figures')
    except:
        pass




# Load LSTM model and scaler
P = 10 # Prediction Horizon
M = 3 # Control Horizon
nCV = 1 # number of CVs
nMV = 1 # number of MVs

# v = load_model('model150.h5')
v = load_model('model_trans150.h5')

multistep = 1

model_params = load(open('model_param_Roaster.pkl', 'rb'))
s1 = model_params['Xscale']
s2 = model_params['yscale']
window = model_params['window']

# Road a data set to get an input sequences
test = pd.read_csv('Roaster_data_random70.csv', index_col=0)

# Number of steps for simulation
nstep = 100
# temporary time array
t = np.linspace(0,nstep+P,nstep+P+1)

# # Simulation Time
tsim = nstep * 10 * 60 # [sec] = nstep * [min/step] * [sec/min] 
# Time interval
tstep = 10 * 60 # sec

# data_input = {
#         "Ore_amps" : np.array(test.Ore_amps[0:nstep]),
#         "Sulfur_tph": np.array(test.Sulfur_tph[0:nstep]),#np.ones(nstep)*test.Sulfur_tph[0],
#         "O2_scfm": np.array(test.O2_scfm[0:nstep]),#np.ones(nstep)*test.O2_scfm[0],
#         "Carbon_in": np.ones(nstep)*test.Carbon_in[0],
#         "Sulf_in": np.ones(nstep)*test.Sulf_in[0],
#         "CO3_in": np.ones(nstep)*test.CO3_in[0],
#         "Gold_in":np.ones(nstep)*0.16,
#         "level1": np.ones(nstep)*140,
#         "level2": np.ones(nstep)*30,
#         "Ore_in_HI": 120,
#         "Ore_in_LO": 98,
#         "Sul_in_HI": 58,
#         "Sul_in_LO": 0,
#         "O2_in_HI": 8000,
#         "O2_in_LO": 4000    
#           }



data_input = {
        "Ore_amps" : np.ones(nstep)*90,
        "Sulfur_tph": np.ones(nstep)*11,
        "O2_scfm": np.ones(nstep)*7400,
        "Carbon_in": np.ones(nstep)*0.87,
        "Sulf_in": np.ones(nstep)*0.64,
        "CO3_in": np.ones(nstep)*17.92,
        "Gold_in":np.ones(nstep)*0.16,
        "level1": np.ones(nstep)*140,
        "level2": np.ones(nstep)*30,
        "Ore_in_HI": 120,
        "Ore_in_LO": 98,
        "Sul_in_HI": 58,
        "Sul_in_LO": 0,
        "O2_in_HI": 8000,
        "O2_in_LO": 4000    
          }

# data_input["Ore_amps"][30:] = 95
# data_input["Ore_amps"][60:] = 85
# data_input["Ore_amps"][100:] = 92
# data_input["Ore_amps"][150:] = 87

# data_input["Sulfur_tph"][30:] = 10.5
# data_input["Sulfur_tph"][60:] = 11.5

# data_input["O2_scfm"][50:] = 7500
# data_input["O2_scfm"][110:] = 7200
# data_input["O2_scfm"][200:] = 7700
# data_input["O2_scfm"][250:] = 7400


sp = np.ones((nstep,nCV))
sp[:,0] = 17 # O2
# sp[:,1] = 45 # CO2
# sp[:,2] = 6 # SO2


# unit conversions to tons/hr
data_tph = {
        "Ore_in": data_input["Ore_amps"] * 1.675786 + 208.270,
        "Ore_in_HI": data_input["Ore_in_HI"] * 1.675786 + 208.270,
        "Ore_in_LO": data_input["Ore_in_LO"] * 1.675786 + 208.270,
        "Sulfur_in": data_input["Sulfur_tph"],
        "Sul_in_HI": data_input["Sul_in_HI"],
        "Sul_in_LO": data_input["Sul_in_LO"],
        "O2_in": data_input["O2_scfm"] * 0.002421,
        "O2_in_HI": data_input["O2_in_HI"] * 0.002421,
        "O2_in_LO": data_input["O2_in_LO"] * 0.002421,
        "Gold_in": 0
          }


data_tph["Gold_in"] = data_input["Gold_in"] * data_tph["Ore_in"]* 2.835e-5 # unit conversion from 'oz/ton' to 'ton/hr'

key_csv_wtpercent_in = {"Carbon_in", "CO3_in", "Sulf_in"}
key_csv_wtpercent_out = {"O2_out", "CO2_out", "SO2_out", "CO3_out", "TCM_out", "Sulf_out"}

# wt% to ton/h
for key in key_csv_wtpercent_in:
    data_tph[key] = data_input[key] * data_tph["Ore_in"] * 1e-2

class input():
    Sulf_in = []
    CO3_in = []
    Carbon_in = []
    Gold_in = []
    Ore_in = []
    Sulfur_in = []
    O2_in = []

# Arrays for storing output data from gekko
O2 = np.zeros(nstep)
CO2 = np.zeros(nstep)
SO2 = np.zeros(nstep)
TCM = np.zeros(nstep)
FeS2 = np.zeros(nstep)
CaCO3 = np.zeros(nstep)
T_1 = np.zeros(nstep)
T_2 = np.zeros(nstep)



class pred():
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
   
# # Arrays for storing prediction data from lstm
# O2_pred = np.zeros(nstep)
# CO2_pred = np.zeros(nstep)
# SO2_pred = np.zeros(nstep)
# TCM_pred = np.zeros(nstep)
# FeS2_pred = np.zeros(nstep)
# CaCO3_pred = np.zeros(nstep)
# T_1_pred = np.zeros(nstep)
# T_2_pred = np.zeros(nstep)

#%% Running process model (Gekko)
# Create plot
nplot = 2
for i in range(nplot):
    plt.figure(i)
    plt.ion()
plt.show()
    
for i in range(0, nstep):
    print('i=',i)
    
    input.Sulf_in = data_tph["Sulf_in"][i]
    input.CO3_in = data_tph["CO3_in"][i]
    input.Carbon_in = data_tph["Carbon_in"][i]
    input.Gold_in = data_tph["Gold_in"][i]
    input.Ore_in = data_tph["Ore_in"][i]
    input.Sulfur_in = data_tph["Sulfur_in"][i]
    input.O2_in = data_tph["O2_in"][i]
    
    output = Gekko_Roaster_model.process(input)
    
    O2[i] = output.O2[-1]
    CO2[i] = output.CO2[-1]
    SO2[i] = output.SO2[-1]
    TCM[i] = output.TCM[-1]
    FeS2[i] = output.FeS2[-1]
    CaCO3[i] = output.CaCO3[-1]
    T_1[i] = output.T_1
    T_2[i] = output.T_2
    
#%% Running LSTM model
    if i <= window:
        pred.Ore_amps = np.ones(P+window)*data_input["Ore_amps"][i]
        pred.Sulfur_tph = np.ones(P+window)*data_input["Sulfur_tph"][i]
        pred.O2_scfm = np.ones(P+window)*data_input["O2_scfm"][i]
        pred.Carbon_in = np.ones(P+window)*data_input["Carbon_in"][i]
        pred.Sulf_in = np.ones(P+window)*data_input["Sulf_in"][i]
        pred.CO3_in = np.ones(P+window)*data_input["CO3_in"][i]
        pred.O2 = np.ones(P+window)*O2[i]
        pred.CO2 = np.ones(P+window)*CO2[i]
        pred.SO2 = np.ones(P+window)*SO2[i]
        pred.TCM = np.ones(P+window)*TCM[i]
        pred.FeS2 = np.ones(P+window)*FeS2[i]
        pred.CaCO3 = np.ones(P+window)*CaCO3[i]
        pred.T_1 = np.ones(P+window)*T_1[i]
        pred.T_2 = np.ones(P+window)*T_2[i]
        
    else:
        
    
        # preparing the input arrays for LSTM prediction
        past = {
                'Ore_amps' : data_input["Ore_amps"][i-window:i],
                'Sulfur_tph' : data_input["Sulfur_tph"][i-window:i], 
                'O2_scfm' : data_input["O2_scfm"][i-window:i],
                'Carbon_in' : data_input["Carbon_in"][i-window:i],
                'Sulf_in' : data_input["Sulf_in"][i-window:i],
                'CO3_in' : data_input["CO3_in"][i-window:i],
                'O2' : O2[i-window:i],
                'CO2' : CO2[i-window:i],
                'SO2' : SO2[i-window:i],
                'TCM' : TCM[i-window:i],
                'FeS2' : FeS2[i-window:i],
                'CaCO3' : CaCO3[i-window:i],
                'T_1' : T_1[i-window:i],
                'T_2' : T_2[i-window:i]
                }
        
        Dof = {
                'Ore_amps' : np.ones(M)*past["Ore_amps"][-1],
                'Sulfur_tph' : np.ones(M)*past["Sulfur_tph"][-1], 
                'O2_scfm' : np.ones(M)*past["O2_scfm"][-1],
                'Carbon_in' : [],
                'Sulf_in' : [],
                'CO3_in' : [],
                'O2' : [],
                'CO2' : [],
                'SO2' : [],
                'TCM' : [],
                'FeS2' : [],
                'CaCO3' : [],
                'T_1' : [],
                'T_2' : []
              }
        
        tail = {
                'Ore_amps' : np.ones(P-M) * Dof["Ore_amps"][-1],
                'Sulfur_tph' : np.ones(P-M) * Dof["Sulfur_tph"][-1],
                'O2_scfm' : np.ones(P-M) * Dof["O2_scfm"][-1],
                'Carbon_in' : np.ones(P) * past["Carbon_in"][-1],
                'Sulf_in' : np.ones(P) * past["Sulf_in"][-1],
                'CO3_in' : np.ones(P) * past["CO3_in"][-1],
                'O2' : np.ones(P) * past["O2"][-1],
                'CO2' : np.ones(P) * past["CO2"][-1],
                'SO2' : np.ones(P) * past["SO2"][-1],
                'TCM' : np.ones(P) * past["TCM"][-1],
                'FeS2' : np.ones(P) * past["FeS2"][-1],
                'CaCO3' : np.ones(P) * past["CaCO3"][-1],
                'T_1' : np.ones(P) * past["T_1"][-1],
                'T_2' : np.ones(P) * past["T_2"][-1]
                }
        
        print('O2=',O2[i])
        print('pred=',pred.O2[0])
        
        
        umeas = {
            'O2': O2[i]-pred.O2[0],
            'CO2': CO2[i]-pred.CO2[0],
            'SO2': SO2[i] - pred.SO2[0],
            'TCM': TCM[i] - pred.TCM[0],
            'FeS2': FeS2[i] - pred.FeS2[0],
            'CaCO3': CaCO3[i] - pred.CaCO3[0],
            'T_1': T_1[i] - pred.T_1[0],
            'T_2': T_2[i] - pred.T_2[0]            
            }
        
        
        if i>20:
            ctl=1 # control switch 0 = off, 1 = on
        else:
            ctl=0
        
        # ctl = 1
        pred = lstm_prediction.rto(past, Dof, tail, window, sp[i,:], v, s1, s2, P, M, ctl, umeas, nCV, nMV,multistep)
        
        if ctl == 1:
            data_input["Ore_amps"][i+1] = pred.Ore_amps[window]
            data_input["Sulfur_tph"][i+1] = pred.Sulfur_tph[window]
            data_input["O2_scfm"][i+1] = pred.O2_scfm[window]
            
            data_tph["Ore_in"][i+1] = data_input["Ore_amps"][i+1] * 1.675786 + 208.270
            data_tph["Sulfur_in"][i+1] = data_input["Sulfur_tph"][i+1]
            data_tph["O2_in"][i+1] = data_input["O2_scfm"][i+1]* 0.002421
        
        
               
    for j in range(nplot):
        plt.figure(j)
        plt.clf()
    
    plt.figure(0)
    plt.subplot(3,1,1)
    plt.plot(t[1:i+1],O2[0:i])
    plt.plot(t[i:i+P],pred.O2[window:window+P], 'r--')
    plt.plot(t[1:i+1],sp[0:i,0], "b--")
    plt.subplot(3,1,2)
    plt.plot(t[1:i+1],CO2[0:i])
    plt.plot(t[i:i+P],pred.CO2[window:window+P], 'r--')
    # plt.plot(t[1:i+1],sp[0:i,1], "b--")
    plt.subplot(3,1,3)
    plt.plot(t[1:i+1],SO2[0:i])
    plt.plot(t[i:i+P],pred.SO2[window:window+P], 'r--')
    # plt.plot(t[1:i+1],sp[0:i,2], "b--")
    plt.draw()
    plt.pause(0.001)
    
    # plt.figure(1)
    # plt.subplot(3,1,1)
    # plt.plot(t[0:i],TCM[0:i], label = 'TCM')
    # plt.plot(t[i:i+P],pred.TCM, 'r--')
    # plt.legend()
    # plt.yticks([])
    # plt.subplot(3,1,2)
    # plt.plot(t[0:i],FeS2[0:i], label = 'FeS2')
    # plt.plot(t[i:i+P],pred.FeS2, 'r--')
    # plt.legend()
    # plt.yticks([])
    # plt.subplot(3,1,3)
    # plt.plot(t[0:i],CaCO3[0:i], label = 'CaCO3')
    # plt.plot(t[i:i+P],pred.CaCO3, 'r--')
    # plt.legend()
    # plt.yticks([])
    # plt.draw()
    # plt.pause(0.1)

    
    
    # plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.plot(t[0:i],T_1[0:i])
    # plt.plot(t[i:i+P],pred.T_1, 'r--')
    # plt.subplot(2,1,2)
    # plt.plot(t[0:i],T_2[0:i])
    # plt.plot(t[i:i+P],pred.T_2, 'r--')
    # plt.draw()
    # # plt.pause(0.001)
    
    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(t[1:i+1],data_input["Ore_amps"][0:i], label = "Ore_amps")
    plt.plot(t[i:i+P],pred.Ore_amps[window:window+P], 'r--') 
    plt.legend()
    # plt.yticks([])
    plt.subplot(3,1,2)
    plt.plot(t[1:i+1],data_input["Sulfur_tph"][0:i], label = "Sulfur")
    plt.plot(t[i:i+P],pred.Sulfur_tph[window:window+P], 'r--')
    plt.legend()
    # plt.yticks([])
    plt.subplot(3,1,3)
    plt.plot(t[1:i+1],data_input["O2_scfm"][0:i], label = "O2")
    plt.plot(t[i:i+P],pred.O2_scfm[window:window+P], 'r--')
    plt.legend()
    # plt.yticks([])
    plt.draw()
    plt.pause(0.1)
    
    if make_mp4:
        filename='./figures/plot_'+str(i+10000)+'.png'
        plt.savefig(filename)
    
    
plt.show()

# generate mp4 from png figures in batches of 350
if make_mp4:
    images = []
    iset = 0
    for i in range(1,nstep):
        filename='./figures/plot_'+str(i+10000)+'.png'
        images.append(imageio.imread(filename))
        if ((i+1)%350)==0:
            imageio.mimsave('results_'+str(iset)+'.mp4', images)
            iset += 1
            images = []
    if images!=[]:
        imageio.mimsave('results_'+str(iset)+'.mp4', images)

