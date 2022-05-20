import warnings

# with warnings.catch_warnings():
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from sklearn import preprocessing

# from functions.process_fopdt import ProcessModel
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
print(s1)
s2 = model_params['yscale']
print(s2)
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

## Set initial temp - both T1 and T2??
T1_set = 35
T2_set = 30

# Setpoint Sequence

sp1 = np.zeros(ns + 1)
sp1 = sp1+T1_set
# sp1[80:] = 1.5

sp2 = np.zeros(ns + 1)
sp2 = sp2+T2_set
# sp2[80:] = 1.5

sp = np.array([sp1, sp2]).T
target = np.array([T1_set,T2_set]).T

# Controller setting
maxmove = 1

# Process simulation
yp = np.zeros((ns + 1, ny))
yp_nn = np.zeros((ns + 1, ny, P))








# p = ProcessModel(delta_t)
m = Mpc_nn(window, nu, ny, P, M, s1, s2, multistep=0, model_one=model_trans_one, model_multi=model_trans_multi)
# multistep = 0 : sequential onestep prediction MPC
# multistep = 1 : simultaneous multistep prediction MPC


print(window)

CI = 30

T1_arr = np.zeros((P*60))
T2_arr = np.zeros((P*60))
Q1_arr = np.zeros((P*60))
Q2_arr = np.zeros((P*60))


tsim = 15*60 # time in sec




## Get initial data
#uhat = np.zeros((M, nu))



lab = tclab.TCLab()


import pickle


print (T1_arr, Q1_arr)
print (T2_arr, Q2_arr)


# # Read data file
# tcL_data = pd.DataFrame(
#         {"H1": Q1_arr,
#          "H2": Q2_arr,
#          "T1": T1_arr,
#          "T2": T2_arr},
#         index = np.linspace(1,P*60,P*60,dtype=int))

# tcL_data.to_pickle('Test1.pkl')



TCL_data = pd.read_pickle('Test1.pkl') # Put the original name


Q1_arr = TCL_data.iloc[:,0]
Q2_arr = TCL_data.iloc[:,1]
T1_arr = TCL_data.iloc[:,2]
T2_arr = TCL_data.iloc[:,3]

print(Q1_arr)
print(Q2_arr)
print(T1_arr)
print(T2_arr)


uhat = np.zeros((M,nu))

Q1_arr = TCL_data.iloc[:,0]
Q2_arr = TCL_data.iloc[:,1]
T1_arr = TCL_data.iloc[:,2]
T2_arr = TCL_data.iloc[:,3]


for i in range(tsim):
    t_in = time.time()
    T1 = lab.T1
    T2 = lab.T2
    Q1 = lab.Q1()
    Q2 = lab.Q2()

    T1_arr = np.append(T1_arr,T1)
    T2_arr = np.append(T2_arr,T2)
    Q1_arr = np.append(Q1_arr,Q1)
    Q2_arr = np.append(Q2_arr,Q2)

    u = np.vstack((Q1_arr,Q2_arr)).T
    y = np.vstack((T1_arr,T2_arr)).T
    target = np.array([T1_set,T2_set]).T
    
    
    if i%30 == 0:
        
        u_window = u[-60*P::30]
        y_window = y[-60*P::30]
        
        uhat = m.run(uhat, u_window, y_window, sp[i])

        lab.Q1(uhat[0][0])
        lab.Q2(uhat[1][0])

    t_out = time.time()
    time.sleep(1-(t_out-t_in))



# plt.plot(t, yp)
plt.subplot(2, 1, 1)
plt.plot(t, y[:, 0])
plt.plot(t, y[:, 1])
# plt.plot(t, yp_nn[:, :, 0], '-.')
plt.step(t, sp[:, 0])
plt.step(t, sp[:, 1])


plt.subplot(2, 1, 2)
plt.step(t, u[:, 0])
plt.step(t, u[:, 1])
plt.show()

print("break point")