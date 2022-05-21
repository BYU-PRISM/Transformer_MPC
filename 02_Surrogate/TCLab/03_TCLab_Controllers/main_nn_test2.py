import sys
import warnings

# with warnings.catch_warnings():
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
from sklearn import preprocessing

# from functions.process_fopdt import ProcessModel
from package.functions.mpc_nn import Mpc_nn

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
# from tclab import TCLabModel as TCLab

TCLab = tclab.setup(connected=False, speedup=10)

# lab = tclab.TCLab()
with TCLab() as lab:

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


    ns = window*30 + 10 * 60  # Simulation Length, min * 60
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
    sp1[window*30:] = 50

    sp2 = np.ones(ns) * sp2_init
    sp2[window*30:] = 30
    
   
    # target = np.array([T1_set,T2_set]).T

    # Controller setting
    maxmove = 1

    # Process simulation
    yp = np.zeros((ns, ny))
    yp_nn = np.zeros((ns, ny, P))

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

    
    # Save python console
    # f = open('console_log.txt', 'w')

    lab.LED(70)


    # Run controller for "ns - len(TCL_data)" sec
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
        # target = np.array([T1_set,T2_set]).T
        
        
        if i%30 == 0:
            
            u_window = u[-30*window::30]
            y_window = y[-30*window::30]
            
            uhat = m.run(uhat, u_window, y_window, sp[i])

            lab.Q1(uhat[0][0])
            lab.Q2(uhat[0][1])
            
            # print(uhat[0], "\n", file=f)
            print(uhat[0], "\n")
        
        sec = i
        Min = sec // 60
        sec %= 60
        # print(Min, "min", sec, "sec   SP1:", sp1[i], " SP2:", sp2[i], 
        #     "   T1:", T1_arr[i], "  T2:", T2_arr[i],
        #     "    H1:", Q1_arr[i], "  H2:", Q2_arr[i], file=f)
        print(Min, "min", sec, "sec   SP1:", sp1[i], " SP2:", sp2[i], 
            "   T1:", T1_arr[i], "  T2:", T2_arr[i],
            "    H1:", Q1_arr[i], "  H2:", Q2_arr[i])
        
        time.sleep(0.1)
        
    lab.LED(0)
    lab.close()

    # Save python console
    # f.close()


    sec = ns
    sec = str(sec)


    # sp_temp = sp[0:400]
    # sp_add = sp[0:ns-400]
    # sp_add1 = np.concatenate((sp_temp,sp_add),axis=0)
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, y[:, 0], 'c-', label='T1')
plt.plot(t, y[:, 1], 'm-', label='T2')
plt.step(t, sp1, 'r--', label='SP1')
plt.step(t, sp2, 'g--', label='SP2')
plt.legend()

plt.subplot(2, 1, 2)
plt.step(t, u[:, 0], 'r-', label='H1')
plt.step(t, u[:, 1], 'g-', label='H2')
plt.legend()

plt.tight_layout()

plt.show()
    # plt.savefig('MPC_Controller_' + sec + 's.png')


    print("break point")