#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings

# with warnings.catch_warnings():
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from sklearn import preprocessing

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



# For LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

import tclab

# Load NN model parameters and MinMaxScaler
model_params = load(open('model_param_MIMO.pkl', 'rb'))
s1 = model_params['Xscale']
s2 = model_params['yscale']
window = model_params['window']

# Load NN models (onestep prediction models)
model_lstm_one = load_model('MPC_MIMO_TCLab_onestep_LSTM.h5')
model_trans_one = load_model('MPC_MIMO_TCLab_onestep_Trans.h5')

# Load NN models (multistep prediction models)
model_lstm_multi = load_model('MPC_MIMO_TCLab_multistep_LSTM.h5')
model_trans_multi = load_model('MPC_MIMO_TCLab_multistep_Trans.h5')

# # FOPDT Parameters
# K=1.0      # gain
# tau=2.0    # time constant
ns = 15*60  # Simulation Length
t = np.linspace(0, ns, ns + 1)
delta_t = t[1] - t[0]

nu = 2
ny = 2

# Define horizons
P = 10  # Prediction Horizon
M = 4  # Control Horizon

# Input Sequence
u = np.zeros((ns + 1, nu))
# u[5:,0] = .3
# u[5:,1] = .6

# Setpoint Sequence

sp1 = np.zeros(ns + 1)
sp1[10:] = 0.5
sp1[40:] = .4
# sp1[80:] = 1.5

sp2 = np.zeros(ns + 1)
sp2[20:] = 0.2
sp2[60:] = 0.1
# sp2[80:] = 1.5

sp = np.array([sp1, sp2]).T

# Controller setting
maxmove = 1

# Process simulation
yp = np.zeros((ns + 1, ny))
yp_nn = np.zeros((ns + 1, ny, P))

# p = ProcessModel(delta_t)
m = Mpc_nn(window, nu, ny, P, M, s1, s2, multistep=1, model_one=model_trans_one, model_multi=model_trans_multi)
# multistep = 0 : sequential onestep prediction MPC
# multistep = 1 : simultaneous multistep prediction MPC




# the control interval in seconds
CI = 30

T1_arr = np.zeros((P*60))
T2_arr = np.zeros((P*60))
Q1_arr = np.zeros((P*60))
Q2_arr = np.zeros((P*60))

## Set initial temp - both T1 and T2??
T1_set = 35
T2_set = 35



tsim = 15*60 # time in sec




## Get initial data
#uhat = np.zeros((M, nu))


lab = tclab.TCLab()


for i in range(len(T1_arr)):
    t_in = time.time()
    T1_arr[i] = lab.T1
    T2_arr[i] = lab.T2
    Q1_arr[i] = lab.Q1()
    Q2_arr[i] = lab.Q2()
    if(T1_arr[i] < T1_set):
        lab.Q1(80)
    else:
        lab.Q1(0)
    if(T2_arr[i] < T2_set):
        lab.Q2(80)
    else:
        lab.Q2(0)
        
    print(T1_arr[i], Q1_arr[i])
    

    t_out = time.time()
    time.sleep(1-(t_out-t_in))


for i in range(tsim):
    t_in = time.time()
    T1 = lab.T1
    T2 = lab.T2
    Q1 = lab.Q1()
    Q2 = lab.Q2()

    T1_arr = np.append(T1_arr,T1)

    #T1_arr.reshape(len(T1_arr)+1)
    #T1_arr[-1] = T1

    T2_arr.reshape(len(T2_arr)+1)
    T2_arr[-1] = T2

    Q1_arr.reshape(len(Q1_arr)+1)
    Q1_arr[-1] = Q1

    Q2_arr.reshape(len(Q2_arr)+1)
    Q2_arr[-1] = Q2

    if i%30 == 0:
        T1_window = T1_arr[-60*P::30]
        Q1_window = Q1_arr[-60*P::30]
        T2_window = T2_arr[-60*P::30]
        Q2_window = Q2_arr[-60*P::30]

        uhat1 = m.run(uhat1, T1_window, Q1_window, T1_set)
        uhat2 = m.run(uhat2, T2_window, Q2_window, T2_set)

        lab.Q1(uhat1[0])
        lab.Q2(uhat2[0])

    t_out = time.time()
    time.sleep(1-(t_out-t_in))



# plt.plot(t, yp)
plt.subplot(2, 1, 1)
plt.plot(t, yp[:, 0])
plt.plot(t, yp[:, 1])
# plt.plot(t, yp_nn[:, :, 0], '-.')
plt.step(t, sp)
plt.subplot(2, 1, 2)
plt.step(t, u[:, 0])
plt.step(t, u[:, 1])
plt.show()

print("break point")


# In[ ]:


import datetime


def convert_seconds_to_time(in_seconds):
    t1   = datetime.timedelta(seconds=in_seconds)
    days = t1.days
    _sec = t1.seconds
    (hours, minutes, seconds) = str(datetime.timedelta(seconds=_sec)).split(':')
    hours   = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    
    result = []
    if days >= 1:
        result.append(str(days)+'d')
    if hours >= 1:
        result.append(str(hours)+'h')
    if minutes >= 1:
        result.append(str(minutes)+'m')
    if seconds >= 1:
        result.append(str(seconds)+'s')
    return ' '.join(result)



nstep = 300

print('Running time: ' + convert_seconds_to_time(nstep) + " (" + str(nstep) + " seconds" + ")" + '\n')

