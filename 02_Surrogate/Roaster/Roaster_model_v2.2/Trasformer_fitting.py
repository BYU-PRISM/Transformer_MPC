import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import time

# For LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Reshape, Conv1DTranspose, Conv2DTranspose

from keras.callbacks import EarlyStopping
from keras.models import load_model
from tqdm.keras import TqdmCallback

window = 6
P = 10
M = 6

# %% Load training data
file = 'Roaster_data_random500.csv'
train = pd.read_csv(file, index_col=0)

train.O2[1:np.shape(train)[0]] = train.O2[0:np.shape(train)[0]-1]
train.CO2[1:np.shape(train)[0]] = train.CO2[0:np.shape(train)[0]-1]
train.SO2[1:np.shape(train)[0]] = train.SO2[0:np.shape(train)[0]-1]
train.TCM[1:np.shape(train)[0]] = train.TCM[0:np.shape(train)[0]-1]
train.FeS2[1:np.shape(train)[0]] = train.FeS2[0:np.shape(train)[0]-1]
train.CaCO3[1:np.shape(train)[0]] = train.CaCO3[0:np.shape(train)[0]-1]
train.T_1[1:np.shape(train)[0]] = train.T_1[0:np.shape(train)[0]-1]
train.T_2[1:np.shape(train)[0]] = train.T_2[0:np.shape(train)[0]-1]

Xm = train[['Ore_amps', 'Sulfur_tph', 'O2_scfm', 'Carbon_in','Sulf_in', 'CO3_in',\
                                'O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']].to_numpy()
Ym = train[['O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']].to_numpy()

nu = 6 # number of input (MV) variables
ny = 8 # number of output (CV) variables

# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
s2 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(Xm)
Ys = s2.fit_transform(Ym)

# Save model parameters
model_params = dict()
model_params['Xscale'] = s1
model_params['Yscale'] = s2
model_params['window'] = window

dump(model_params, open('model_param_Roaster.pkl', 'wb'))


# Snapshots for multistep model
X = []
Y = []
for i in range(window,len(Xs)-P):
    X.append(Xs[i-window:i+P,:])
    Y.append(Ys[i:i+P])


# Reshape data to format accepted by LSTM
X, Y = np.array(X), np.array(Y)

# Modifying "yhat" part of data for multistep model
for i in range(0, len(X)):
    for j in range(nu, nu+ny):
        X[i][window:,j] = X[i][window-1,j]
       
# create and train transformer model

# Initialize LSTM model
n_feature = X.shape[2]
n_label = Y.shape[1]

encoder_input = tf.keras.layers.Input(shape=(window+P, n_feature))
xl = tf.keras.layers.MultiHeadAttention(num_heads=10,key_dim=2)(encoder_input,encoder_input)
res = xl + encoder_input
xl = tf.keras.layers.Dense(100, 'tanh')(res)
xl = tf.keras.layers.Dropout(0.2)(xl)
xl = tf.keras.layers.Dense(n_feature, None)(xl)

xl = tf.keras.layers.MultiHeadAttention(num_heads=10,key_dim=2)(xl,xl)
res = xl + encoder_input
xl = tf.keras.layers.Dense(100, 'tanh')(res)
xl = tf.keras.layers.Dropout(0.2)(xl)
xl = tf.keras.layers.Dense(n_feature, None)(xl)

xl = tf.keras.layers.Flatten()(xl)
xl = tf.keras.layers.Dense(units=Y.shape[1]*Y.shape[2], activation=None)(xl)
yl = tf.keras.layers.Reshape((Y.shape[1], Y.shape[2]))(xl)

model_trans = tf.keras.Model(inputs=encoder_input, outputs=yl, name="mhl")
model_trans.compile(optimizer='Adam',loss='MSE',metrics='accuracy')

# Allow for early exit
es_trans = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# t0 = time.time()
# history = model_trans.fit(X,Y,epochs=300,batch_size=32, \
#                                callbacks=[es_trans, TqdmCallback(verbose=1)], \
#                                verbose=0)
# t1 = time.time()

# print('Runtime: %.2f s' %(t1-t0))

# model_trans.summary()

# # Plot loss
# plt.figure(figsize=(8,4))
# plt.semilogy(history.history['loss'])
# plt.xlabel('epoch'); plt.ylabel('loss')
# plt.savefig('tclab_loss.png')
# model_trans.save('model_trans150.h5')


model_trans = load_model('model_trans15000.h5')

# Verify the fit of the model
Yp = model_trans.predict(X)

# Ypu = s2.inverse_transform(Yp)

Ypu = (Yp - s2.min_) * 1/s2.scale_ + s2.data_min_ 
Yu = (Y - s2.min_) * 1/s2.scale_ + s2.data_min_ 

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Yp[0:300,5,0],label='Transformer')
# plt.plot(Y[0:300,5,0],"--", label="Meas")
# plt.legend()

# plt.subplot(3,1,2)
# plt.plot(Yp[0:300,5,1],label='Transformer')
# plt.plot(Y[0:300,5,1],"--", label="Meas")
# plt.legend()

# plt.subplot(3,1,3)
# plt.plot(Y[0:300,5,1],label='Transformer')
# plt.plot(Y[0:300,5,1],"--", label="Meas")
# plt.legend()

# plt.show()

# Plotting for 

t=np.linspace(0,len(Yp)-1,len(Yp))



begin = 0
end = 1500
skip = P-1


plt.style.use('seaborn-white')
plt.figure(0,figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t[begin:begin+P], Ypu[begin][:,0], '-', color="red",linewidth=1, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,0], '--', color="blue",linewidth=2, label="Measured")
for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,0], '-',color="red",linewidth=1)
    plt.plot(t[i:i+P], Yu[i][:,0], '--', color="blue",linewidth=2)
    
plt.title('Transformer Multistep Model Validation - PINN Mode Off',fontsize=15)
plt.ylabel("y1", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)


plt.subplot(2,1,2)
plt.plot(t[begin:begin+P], Ypu[begin][:,1], '-',color="red",linewidth=1, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,1], '--', color="blue",linewidth=2, label="Measured")
for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,1], '-',color="red",linewidth=1)
    plt.plot(t[i:i+P], Yu[i][:,1], '--', color="blue",linewidth=2)
    
plt.tight_layout()
plt.ylabel("y2", fontsize=14)
plt.xlabel("Time (Seconds)", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)
plt.tight_layout()

# plt.savefig('TCLab_Training_Trans_multi_PINN_Off.eps', format='eps')
# plt.savefig('TCLab_Training_Trans_multi_PINN_Off.png', format='png')



plt.figure(1, figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t[begin:begin+P], Ypu[begin][:,3], '-',color="red",linewidth=2, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,3], '--',color="blue", linewidth=2, label="Measured")

for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,3], '-',color="red",linewidth=2)
    plt.plot(t[i:i+P], Yu[i][:,3], '--', color="blue",linewidth=2)
    
plt.ylabel("y1", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)
plt.title('Transformer Multistep Model Validation - PINN Mode On',fontsize=15)    


plt.subplot(2,1,2)
plt.plot(t[begin:begin+P], Ypu[begin][:,4], '-',color="red",linewidth=2, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,4], '--',color="blue", linewidth=2, label="Measured")

for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,4], '-',color="red",linewidth=2)
    plt.plot(t[i:i+P], Yu[i][:,4], '--',color="blue", linewidth=2)
    

plt.ylabel("y2", fontsize=14)
plt.xlabel("Time (Seconds)", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)
plt.tight_layout() 


plt.figure(3, figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t[begin:begin+P], Ypu[begin][:,5], '-',color="red",linewidth=2, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,5], '--',color="blue", linewidth=2, label="Measured")

for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,5], '-',color="red",linewidth=2)
    plt.plot(t[i:i+P], Yu[i][:,5], '--', color="blue",linewidth=2)
    
plt.ylabel("y1", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)
plt.title('Transformer Multistep Model Validation - PINN Mode On',fontsize=15)    


plt.subplot(2,1,2)
plt.plot(t[begin:begin+P], Ypu[begin][:,6], '-',color="red",linewidth=2, label="Prediction")
plt.plot(t[begin:begin+P], Yu[begin][:,6], '--',color="blue", linewidth=2, label="Measured")

for i in range(begin,end,skip):   
    plt.plot(t[i:i+P], Ypu[i][:,6], '-',color="red",linewidth=2)
    plt.plot(t[i:i+P], Yu[i][:,6], '--',color="blue", linewidth=2)
    

plt.ylabel("y2", fontsize=14)
plt.xlabel("Time (Seconds)", fontsize=14)
plt.legend(loc=2,fontsize=14)
plt.tick_params(axis='both',labelsize=14)
plt.tight_layout() 
    
# plt.savefig('TCLab_Training_Trans_multi_PINN_On.eps', format='eps')
# plt.savefig('TCLab_Training_Trans_multi_PINN_On.png', format='png')
    

plt.show()



# Xs[window:,nu:] = Yp

# # un-scale outputs
# Xu = s1.inverse_transform(Xs)


# plt.subplot(3,1,2)
# plt.plot(Yu[:,1],'r-',label='LSTM')
# plt.plot(Ym[:,1],'b--',label='Measured')
# plt.legend()
# plt.xlabel('Time (sec)');

# plt.subplot(3,1,3)
# plt.plot(Yu[:,2],'r-',label='LSTM')
# plt.plot(Ym[:,2],'b--',label='Measured')

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Yu[:,3],'r-',label='LSTM')
# plt.plot(Ym[:,3],'b--',label='Measured')

# plt.subplot(3,1,2)
# plt.plot(Yu[:,4],'r-',label='LSTM')
# plt.plot(Ym[:,4],'b--',label='Measured')

# plt.subplot(3,1,3)
# plt.plot(Yu[:,5],'r-',label='LSTM')
# plt.plot(Ym[:,5],'b--',label='Measured')

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(Xu[:,6],'r-',label='LSTM')
# plt.plot(Xm[:,6],'b--',label='Measured')

# plt.subplot(2,1,2)
# plt.plot(Xu[:,7],'r-',label='LSTM')
# plt.plot(Xm[:,7],'b--',label='Measured')

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(Xu[:,13],'r-',label='LSTM')
# plt.plot(Xm[:,13],'b--',label='Measured')

# plt.subplot(2,1,2)
# plt.plot(Xu[:,13],'r-',label='LSTM')
# plt.plot(Xm[:,13],'b--',label='Measured')
# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(train['Time'][window:],Yu[:,0],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,0],'b--',label='Measured')
# plt.legend()

# plt.subplot(3,1,2)
# plt.plot(train['Time'][window:],Yu[:,1],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,1],'b--',label='Measured')
# plt.legend()
# plt.xlabel('Time (sec)');

# plt.subplot(3,1,3)
# plt.plot(train['Time'][window:],Yu[:,2],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,2],'b--',label='Measured')

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(train['Time'][window:],Yu[:,3],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,3],'b--',label='Measured')

# plt.subplot(3,1,2)
# plt.plot(train['Time'][window:],Yu[:,4],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,4],'b--',label='Measured')

# plt.subplot(3,1,3)
# plt.plot(train['Time'][window:],Yu[:,5],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,5],'b--',label='Measured')

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(train['Time'][window:],Yu[:,6],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,6],'b--',label='Measured')

# plt.subplot(2,1,2)
# plt.plot(train['Time'][window:],Yu[:,7],'r-',label='LSTM')
# plt.plot(train['Time'][window:],Ym[:,7],'b--',label='Measured')

# #%% Load model
# v = load_model('model150.h5')

# model_params = load(open('model_param_Roaster.pkl', 'rb'))
# s1 = model_params['Xscale']
# window = model_params['window']


# # # Load training data
# test = pd.read_csv('Roaster_data_random30.csv', index_col=0)
# # # test = test[0::5]
# nstep = test.shape[0]

# test.O2[1:np.shape(test)[0]] = test.O2[0:np.shape(test)[0]-1]
# test.CO2[1:np.shape(test)[0]] = test.CO2[0:np.shape(test)[0]-1]
# test.SO2[1:np.shape(test)[0]] = test.SO2[0:np.shape(test)[0]-1]
# test.TCM[1:np.shape(test)[0]] = test.TCM[0:np.shape(test)[0]-1]
# test.FeS2[1:np.shape(test)[0]] = test.FeS2[0:np.shape(test)[0]-1]
# test.CaCO3[1:np.shape(test)[0]] = test.CaCO3[0:np.shape(test)[0]-1]
# test.T_1[1:np.shape(test)[0]] = test.T_1[0:np.shape(test)[0]-1]
# test.T_2[1:np.shape(test)[0]] = test.T_2[0:np.shape(test)[0]-1]

# Xtm = np.array(test)

# Xt = test[['Ore_amps', 'Sulfur_tph', 'O2_scfm', 'Carbon_in','Sulf_in', 'CO3_in',\
#            'O2', 'CO2', 'SO2', 'TCM', 'FeS2', 'CaCO3', 'T_1', 'T_2']]

# Xts = s1.transform(Xt)

# #%% Using predicted values to predict next step

# Xtp = np.zeros([nstep-window,14])

# for i in range(window,nstep):
#     Xin = Xts[i-window:i].reshape((1, window, 14))
#     Xts[i][6:] = v.predict(Xin)
#     Xtp[i-window] = Xts[i]
    

# Xtu = s1.inverse_transform(Xts)

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Xtu[:,6],'r-',label='LSTM')
# plt.plot(Xtm[:,6],'b--',label='Measured')
# plt.legend()
# plt.yticks([])


# plt.subplot(3,1,2)
# plt.plot(Xtu[:,7],'r-',label='LSTM')
# plt.plot(Xtm[:,7],'b--',label='Measured')
# plt.legend()
# plt.xlabel('Time (sec)')
# plt.yticks([])

# plt.subplot(3,1,3)
# plt.plot(Xtu[:,8],'r-',label='LSTM')
# plt.plot(Xtm[:,8],'b--',label='Measured')
# plt.yticks([])

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(Xtu[:,9],'r-',label='LSTM')
# plt.plot(Xtm[:,9],'b--',label='Measured')
# plt.yticks([])

# plt.subplot(3,1,2)
# plt.plot(Xtu[:,10],'r-',label='LSTM')
# plt.plot(Xtm[:,10],'b--',label='Measured')
# plt.yticks([])

# plt.subplot(3,1,3)
# plt.plot(Xtu[:,11],'r-',label='LSTM')
# plt.plot(Xtm[:,11],'b--',label='Measured')
# plt.yticks([])

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(Xtu[:,12],'r-',label='LSTM')
# plt.plot(Xtm[:,12],'b--',label='Measured')
# plt.yticks([])

# plt.subplot(2,1,2)
# plt.plot(Xtu[:,13],'r-',label='LSTM')
# plt.plot(Xtm[:,13],'b--',label='Measured')
# plt.yticks([])

# plt.figure()
# plt.scatter(Xtm[:,13], Xtu[:,13])
# z = np.polyfit(Xtm[:,13], Xtu[:,13], 1)
# plt.ylabel("LSTM")
# plt.xlabel("Measured")
# p = np.poly1d(z)
# plt.plot(Xtm[:,13],p(Xtm[:,13]),"r--")
# plt.yticks([])
# plt.xticks([])

# plt.show()

# R2 = r2_score(Xtu[:,13], p(Xtm[:,13]))
# print(R2)

