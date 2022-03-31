from functions.process_fopdt import ProcessModel
from functions.mpc_nn import Mpc_nn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import time

path = 'data/'

# For LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load NN model parameters and MinMaxScaler
model_params = load(open(path +'model_param_MIMO.pkl', 'rb'))
s1 = model_params['Xscale']
s2 = model_params['yscale']
window = model_params['window']

# Load NN models (onestep prediction models)
model_lstm_one = load_model(path +'MPC_MIMO_FOPDT_onestep_LSTM.h5')
model_trans_one = load_model(path +'MPC_MIMO_FOPDT_onestep_Trans.h5')

# Load NN models (multistep prediction models)
model_lstm_multi = load_model(path +'MPC_MIMO_FOPDT_multistep_LSTM.h5')
model_trans_multi = load_model(path +'MPC_MIMO_FOPDT_multistep_Trans.h5')

# # FOPDT Parameters
# K=1.0      # gain
# tau=2.0    # time constant
ns = 20    # Simulation Length
t = np.linspace(0,ns,ns+1)
delta_t = t[1]-t[0]

nu = 2
ny = 2

# Define horizons
P = 10 # Prediction Horizon
M = 4  # Control Horizon

# Input Sequence
u = np.zeros((ns+1,nu))
# u[5:,0] = 5
# u[10:,1] = 3

# Setpoint Sequence

sp1 = np.zeros(ns+1)
sp1[10:40] = 2
sp1[40:] = 1
# sp1[80:] = 1.5

sp2 = np.zeros(ns+1)
sp2[10:40] = 1
sp2[60:] = 2
# sp2[80:] = 1.5

sp = np.array([sp1, sp2]).T


# Controller setting
maxmove = 1

## Process simulation 
yp = np.zeros((ns+1,ny))

p = ProcessModel(delta_t)
m = Mpc_nn(window, nu, ny, P, M, s1, s2, multistep=0, model_one=model_lstm_one, model_multi=model_lstm_multi)
# multistep = 0 : sequential onestep prediction MPC
# multistep = 1 : simultaneous multistep prediction MPC

uhat = np.zeros((M, nu))
for i in range(1, window):
    yp[i] = p.run(u[i-1])

u_window = u[0:window]
y_window = yp[0:window]

for i in range(window,ns):
    print(i)
    # run process model
    yp[i+1] = p.run(u[i])

    # run MPC 
    uhat = m.run(uhat, u_window, y_window, sp[i])
    u[i+1] = uhat[0]
    # delta = u[i+1] - u[i]
    
    # if np.abs(delta) >= maxmove:
    #     if delta > 0:
    #         u[i+1] = u[i]+maxmove
    #     else:
    #         u[i+1] = u[i]-maxmove


    print(uhat[0])
    u_window = u[i-window+1:i+1]
    y_window = yp[i-window+1:i+1]

    
# plt.plot(t, yp)
plt.subplot(2,1,1)
plt.plot(t, yp[:,0])
plt.plot(t, yp[:,1])
plt.step(t, sp)
plt.subplot(2,1,2)
plt.step(t, u[:,0])
plt.step(t,u[:,1])
plt.show()
 
print("break point")