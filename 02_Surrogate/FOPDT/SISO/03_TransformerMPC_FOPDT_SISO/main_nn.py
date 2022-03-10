from process_fopdt import *
from mpc_lstm import Mpc

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import time

path = '../'

# For LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

# Load NN model parameters and MinMaxScaler
model_params = load(open(path +'model_param.pkl', 'rb'))
s1 = model_params['Xscale']
s2 = model_params['yscale']
window = model_params['window']

# Load NN models (onestep prediction models)
model_lstm = load_model(path +'MPC_SISO_FOPDT_onestep_LSTM.h5')
# model_trans = load_model(path +'MPC_SISO_FOPDT_onestep_Trans.h5')

# Load NN models (multistep prediction models)
model_lstm_multi = load_model(path +'MPC_SISO_FOPDT_multistep_LSTM.h5')
# model_trans_multi = load_model(path +'MPC_SISO_FOPDT_multistep_Trans.h5')

# FOPDT Parameters
K=1.0      # gain
tau=2.0    # time constant
ns = 100    # Simulation Length
t = np.linspace(0,ns,ns+1)
delta_t = t[1]-t[0]


# Define horizons
P = 10 # Prediction Horizon
M = 4  # Control Horizon

# Input Sequence
u = np.zeros(ns+1)
# u[5:] = 5

# Setpoint Sequence

sp = np.zeros(ns+1)
sp[10:40] = 2
sp[40:80] = 1
sp[80:] = 1.5
# Controller setting
maxmove = 1

## Process simulation 
yp = np.zeros(ns+1)

p = ProcessModel(K, tau, delta_t)
m = Mpc(window, P, M, s1, s2, multistep=1, model=model_lstm, model_multi=model_lstm_multi)
# multistep = 0 : sequential onestep prediction MPC
# multistep = 1 : simultaneous multistep prediction MPC

uhat = np.zeros(M)
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
    delta = u[i+1] - u[i]
    
    if np.abs(delta) >= maxmove:
        if delta > 0:
            u[i+1] = u[i]+maxmove
        else:
            u[i+1] = u[i]-maxmove


    u_window = u[i-window+1:i+1]
    y_window = yp[i-window+1:i+1]

    
plt.plot(t, yp)
plt.step(t, u)
plt.step(t, sp)
plt.show()
 




