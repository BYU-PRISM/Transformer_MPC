# import sys
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

lab = tclab.TCLab()

# Make an MP4 animation?
make_mp4 = False
if make_mp4:
    import imageio  # required to make animation
    import os
    try:
        os.mkdir('./figures')
    except:
        pass

# Load NN model parameters and MinMaxScaler
model_params = load(open('model_param_mimo_TCLab.pkl', 'rb'))
s1 = model_params['Xscale']
s2 = model_params['yscale']
window = model_params['window']

# Load NN models (onestep prediction models)
model_lstm_one = load_model('PINN_TCLab_mimo_multistep_Trans_pinn_on.h5')
model_trans_one = load_model('PINN_TCLab_mimo_multistep_Trans_pinn_on.h5')

# Load NN models (multistep prediction models)
model_lstm_multi = load_model('PINN_TCLab_mimo_multistep_Trans_pinn_off.h5')
model_trans_multi = load_model('PINN_TCLab_mimo_multistep_Trans_pinn_off.h5')


ns = 60 * 60  # Simulation Length, min * 60
t = np.linspace(0, ns-1, ns)

nu = 2
ny = 2

# Define horizons
P = 10  # Prediction Horizon
M = 4  # Control Horizon

# Input Sequence
# u = np.zeros((ns, nu))

## Set initial set_temp
sp1_init = lab.T1
sp2_init = lab.T2

# Setpoint Sequence
sp1 = np.ones(ns) * sp1_init
sp1[window*30-1:] = 43
sp1[10*60:] = 53
sp1[16*60:] = 62
sp1[25*60:] = 74
sp1[36*60:] = 85
sp1[47*60:] = 70
sp1[58*60:] = 65
sp1[75*60:] = 47
sp1[93*60:] = 42
sp1[108*60:] = 33



sp2 = np.ones(ns) * sp2_init
sp2[window*30-1:] = 35
sp2[13*60:] = 40
sp2[19*60:] = 59
sp2[30*60:] = 65
sp2[50*60:] = 52
sp2[58*60:] = 45
sp2[65*60:] = 65
sp2[73*60:] = 35
sp2[91*60:] = 39
sp2[104*60:] = 31


# Controller setting
maxmove = 1

# Process simulation
yp = np.zeros((ns, ny))
yp_nn = np.zeros((ny, ))

# m = Mpc_nn(window, nu, ny, P, M, s1, s2, multistep=0, model_one=model_lstm_one, model_multi=model_lstm_multi)
m = Mpc_nn(window, nu, ny, P, M, s1, s2, multistep=1, model_one=model_trans_one, model_multi=model_trans_multi)



# Set initial fake data
TCL_data = pd.read_pickle('initial_fake_data.pkl')
TCL_data = TCL_data[0:window*30]

Q1_arr = TCL_data.iloc[:,0]
Q2_arr = TCL_data.iloc[:,1]
T1_arr = TCL_data.iloc[:,2]
T2_arr = TCL_data.iloc[:,3]

T1_arr = T1_arr + (lab.T1 - np.average(T1_arr))
T2_arr = T2_arr + (lab.T2 - np.average(T2_arr))

uhat = np.zeros((M,nu))

elapsed = np.zeros(ns)

lab.LED(70)

# Create plot
nplot = 1
for i in range(nplot):
    plt.figure(i)
    plt.ion()
plt.show()

for i in range(30*window, ns):

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
    sp = np.array([sp1, sp2]).T

    if i%10 == 0:
        
        u_window = u[-30*(window-1)-1::30]
        y_window = y[-30*(window-1)-1::30]
        
        if i == window *30: # to exclude ffwd for the first mpc run (bc there is no y_nn)
            ffwd = 0
        else:    
            ffwd = np.array([T1,T2]) - yp_nn 
        
        start = time.time()
        uhat = m.run(uhat, u_window, y_window, sp[i], ffwd)
        end = time.time()
        elapsed[i] = end - start
        print(elapsed[i])

        lab.Q1(uhat[0][0])
        lab.Q2(uhat[0][1])

        yp_nn = m.RunNN(uhat, u_window, y_window, sp[i])[0]
        
        
        print(uhat[0], "\n")
    
    sec = i
    Min = sec // 60
    sec %= 60
    print(Min, "min", sec, "sec   SP1:", sp1[i], " SP2:", sp2[i], 
        "   T1:", T1_arr[i], "  T2:", T2_arr[i],
        "    H1:", Q1_arr[i], "  H2:", Q2_arr[i])
    
    time.sleep(1)

    for j in range(nplot):
        plt.figure(j)
        plt.clf()
    
    plt.figure(0)
    plt.subplot(2,1,1)
    plt.plot(t[0:i+1],T1_arr, label = "T1")
    plt.plot(t[0:i+1],sp1[0:i+1], 'r--',label = "SP1")
    plt.plot(t[0:i+1],T2_arr, label = "T2")
    plt.plot(t[0:i+1],sp2[0:i+1], 'r--',label = "SP2")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(t[0:i+1],Q1_arr, label = "Q1")
    plt.plot(t[0:i+1],Q2_arr, label = "Q2")
    plt.legend()
    plt.draw()
    plt.pause(0.1)

    if make_mp4:
        filename='./figures/plot_'+str(i+10000)+'.png'
        plt.savefig(filename)

plt.savefig('TCLab_PINN_Off_MIMO_Control_multi_Trans_20_20_1_1_1hr.png')
plt.savefig('TCLab_PINN_Off_MIMO_Control_multi_Trans_20_20_1_1_1hr.eps', format='eps')
plt.show()

lab.LED(0)
lab.close()


# Read data file
tcL_data = pd.DataFrame(
        {"H1": Q1_arr,
         "H2": Q2_arr,
         "T1": T1_arr,
         "T2": T2_arr,
         "SP1": sp1,
         "SP2": sp2,
         "elapsed":elapsed},
        index = np.linspace(1,ns,ns,dtype=int))

tcL_data.to_pickle('TCLab_PINN_Off_MIMO_Control_Multi_Trans_20_20_1_1_1hr.pkl')

# generate mp4 from png figures in batches of 350
if make_mp4:
    images = []
    iset = 0
    for i in range(window*30,ns):
        filename='./figures/plot_'+str(i+10000)+'.png'
        images.append(imageio.imread(filename))
        if ((i+1)%1000)==0:
            imageio.mimsave('results_'+str(iset)+'.mp4', images)
            iset += 1
            images = []
    if images!=[]:
        imageio.mimsave('results_'+str(iset)+'.mp4', images)
    


