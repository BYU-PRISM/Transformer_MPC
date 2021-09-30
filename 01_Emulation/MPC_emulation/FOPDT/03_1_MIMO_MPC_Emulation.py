# from MIMO_FOPDT_MPC_data_creating import SP1
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from pickle import dump, load
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
import joblib

from tqdm import tqdm # Progress bar

# Load NN model parameters
model_params = load(open('model_param.pkl', 'rb'))

s_x = model_params['Xscale']
s_y = model_params['yscale']
window = model_params['window']

tf = 150 # fianl time
# Setpoint Scenario for closed-loop
SP1 = np.zeros(tf)
SP2 = np.zeros(tf)

SP1[10:] = 0.5
SP2[10:] = 0.2
SP1[30:] = 0.3
SP2[50:] = 0.5
SP1[80:] = 0.6
SP2[100:] = 0.4
SP1[120:] = 0.8
SP2[140:] = 0.5


# Plant model

p=GEKKO(remote=True)
p.time = [0, 1]

# Process Gain
p.K11 = p.FV(2)
p.K12 = p.FV(.5) 
p.K21 = p.FV(.5) 
p.K22 = p.FV(2)

# Time Constant
p.tau11 = p.FV(5)
p.tau12 = p.FV(5) 
p.tau21 = p.FV(5) 
p.tau22 = p.FV(5) 

# Gekko variables for Input Output
p.y1 = p.Var(0)
p.y2 = p.Var(0)
p.u1 = p.Param(0, lb=0, ub=5)
p.u2 = p.Param(0, lb=0, ub=5)

#FOPDT Equation
p.Equation(p.y1.dt()+p.y1 == p.K11/p.tau11*p.u1 + p.K21/p.tau21*p.u2) 
p.Equation(p.y2.dt()+p.y2 == p.K12/p.tau12*p.u1 + p.K22/p.tau22*p.u2) 

p.options.IMODE = 4

##---------------------------------
##    MPC model (Gekko)
##----------------------------------
P = 20 # Prediction Horizon
m=GEKKO(remote=True)
m.time = np.linspace(0,P-1,P)

# Process Gain
m.K11 = m.FV(2)
m.K12 = m.FV(.5) 
m.K21 = m.FV(.5) 
m.K22 = m.FV(2)

# Time Constant
m.tau11 = m.FV(5)
m.tau12 = m.FV(5) 
m.tau21 = m.FV(5) 
m.tau22 = m.FV(5) 

# # Input Scenario for open-loop
# # u1_input = np.zeros(tf)
# # u1_input[5:] = 1
# # u2_input = np.zeros(tf)
# # u2_input[15:] = -1

# Gekko variables for Input Output
m.y1 = m.CV(0)
m.y2 = m.CV(0)
m.u1 = m.MV(0, lb=0, ub=5)
m.u2 = m.MV(0, lb=0, ub=5)

m.y1.STATUS = 1
m.y2.STATUS = 1
m.y1.FSTATUS = 1
m.y2.FSTATUS = 1

m.u1.STATUS = 1
m.u2.STATUS = 1
m.u1.FSTATUS = 0
m.u2.FSTATUS = 0

m.u1.DCOST = 1e5
m.u2.DCOST = 1e5
m.y1.WSP = 1e6
m.y2.WSP = 1e6

#FOPDT Equation
m.Equation(m.y1.dt()+m.y1 == m.K11/m.tau11*m.u1 + m.K21/m.tau21*m.u2) 
m.Equation(m.y2.dt()+m.y2 == m.K12/m.tau12*m.u1 + m.K22/m.tau22*m.u2) 

m.options.CV_TYPE = 2
m.options.IMODE = 6

# Storage MPC control result 
u1_mpc = np.ones(tf)*p.u1.VALUE
u2_mpc = np.ones(tf)*p.u2.VALUE
y1_mpc = np.ones(tf)*p.y1.VALUE
y2_mpc = np.ones(tf)*p.y2.VALUE

for i in range(tf-1):
    p.u1.VALUE = u1_mpc[i]
    p.u2.VALUE = u2_mpc[i]
 
    p.solve(disp=False)

    y1_mpc[i+1] = p.y1.VALUE[-1]
    y2_mpc[i+1] = p.y2.VALUE[-1]

    m.y1.MEAS = y1_mpc[i+1]
    m.y2.MEAS = y2_mpc[i+1]

    m.solve(disp=False)

    m.y1.SP = SP1[i+1]
    m.y2.SP = SP2[i+1]
    
    u1_mpc[i+1] = m.u1.NEWVAL
    u2_mpc[i+1] = m.u2.NEWVAL

u1_mpc = u1_mpc[10:]
u2_mpc = u2_mpc[10:]
y1_mpc = y1_mpc[10:]
y2_mpc = y2_mpc[10:]

plt.figure(0)
plt.subplot(2,1,1)
plt.plot(SP1[window:], drawstyle='steps')
plt.plot(SP2[window:], drawstyle='steps')
plt.plot(y1_mpc)
plt.plot(y2_mpc)
plt.legend(['SP1', 'SP2', 'y1', 'y2'])

plt.subplot(2,1,2)
plt.plot(u1_mpc, drawstyle='steps')
plt.plot(u2_mpc, drawstyle='steps')
plt.legend(['u1', 'u2'])
plt.show()


#-----------------------------------------
# LSTM Controller emulation
#-----------------------------------------
model_LSTM = load_model('MPC_emulate_LSTM.h5')
model_params = load(open('model_param.pkl', 'rb'))

s_x = model_params['Xscale']
s_y = model_params['yscale']
window = model_params['window']


p.u1.VALUE = 0
p.u2.VALUE = 0
p.y1.VALUE = 0
p.y2.VALUE = 0

# Storage LSTM control result 
u1_lstm = np.ones(tf)*p.u1.VALUE
u2_lstm = np.ones(tf)*p.u2.VALUE
y1_lstm = np.ones(tf)*p.y1.VALUE
y2_lstm = np.ones(tf)*p.y2.VALUE


for i in range(tf-1):
    p.u1.VALUE = u1_lstm[i]
    p.u2.VALUE = u2_lstm[i]
 
    p.solve(disp=False)

    y1_lstm[i+1] = p.y1.VALUE[-1]
    y2_lstm[i+1] = p.y2.VALUE[-1]

    print("break")
    
    if i >= window:
        input_y1 = y1_lstm[i-window:i]
        input_y2 = y2_lstm[i-window:i]
        input_sp1 = SP1[i-window:i]
        input_sp2 = SP2[i-window:i]

        X = np.vstack((input_y1,input_y2,input_sp1,input_sp2)).T
       
        Xs = s_x.transform(X)

        Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

        Ys = model_LSTM.predict(Xs)

        u1_lstm[i+1], u2_lstm[i+1] = s_y.inverse_transform(Ys)[0]

        if u1_lstm[i+1] < 0:
            u1_lstm[i+1] = 0
        if u2_lstm[i+1] < 0:
            u2_lstm[i+1] = 0
        if u1_lstm[i+1] > 100:
            u1_lstm[i+1] = 100
        if u2_lstm[i+1] > 100:
            u2_lstm[i+1] = 100

plt.figure(0)
plt.subplot(2,1,1)
plt.plot(SP1[window:], drawstyle='steps')
plt.plot(SP2[window:], drawstyle='steps')
plt.plot(y1_lstm[window+1:])
plt.plot(y2_lstm[window+1:])
plt.legend(['SP1', 'SP2', 'y1', 'y2'])

plt.subplot(2,1,2)
plt.plot(u1_lstm[window+1:], drawstyle='steps')
plt.plot(u2_lstm[window+1:], drawstyle='steps')
plt.legend(['u1', 'u2'])
plt.show()


# #-----------------------------------------
# # Transformer Controller emulation
# #-----------------------------------------
model_Trans = load_model('MPC_emulate_Transformer.h5')
model_params = load(open('model_param.pkl', 'rb'))

s_x = model_params['Xscale']
s_y = model_params['yscale']
window = model_params['window']

p.u1.VALUE = 0
p.u2.VALUE = 0
p.y1.VALUE = 0
p.y2.VALUE = 0

# Storage LSTM control result 
u1_trans = np.ones(tf)*p.u1.VALUE
u2_trans = np.ones(tf)*p.u2.VALUE
y1_trans = np.ones(tf)*p.y1.VALUE
y2_trans = np.ones(tf)*p.y2.VALUE


for i in range(tf-1):
    p.u1.VALUE = u1_trans[i]
    p.u2.VALUE = u2_trans[i]
 
    p.solve(disp=False)

    y1_trans[i+1] = p.y1.VALUE[-1]
    y2_trans[i+1] = p.y2.VALUE[-1]

    print("break")
    
    if i >= window:
        input_y1 = y1_trans[i-window:i]
        input_y2 = y2_trans[i-window:i]
        input_sp1 = SP1[i-window:i]
        input_sp2 = SP2[i-window:i]

        X = np.vstack((input_y1,input_y2,input_sp1,input_sp2)).T
       
        Xs = s_x.transform(X)

        Xs = np.reshape(Xs, (1, Xs.shape[0], Xs.shape[1]))

        Ys = model_Trans.predict(Xs)

        u1_trans[i+1], u2_trans[i+1] = s_y.inverse_transform(Ys)[0]

        if u1_trans[i+1] < 0:
            u1_trans[i+1] = 0
        if u2_trans[i+1] < 0:
            u2_trans[i+1] = 0
        if u1_trans[i+1] > 100:
            u1_trans[i+1] = 100
        if u2_trans[i+1] > 100:
            u2_trans[i+1] = 100


plt.figure(0)
plt.subplot(2,1,1)
plt.plot(SP1[window:], drawstyle='steps')
plt.plot(SP2[window:], drawstyle='steps')
plt.plot(y1_trans[window+1:])
plt.plot(y2_trans[window+1:])
plt.legend(['SP1', 'SP2', 'y1', 'y2'])

plt.subplot(2,1,2)
plt.plot(u1_trans[window+1:], drawstyle='steps')
plt.plot(u2_trans[window+1:], drawstyle='steps')
plt.legend(['u1', 'u2'])
plt.show()


print("break")
        

file = open(file='MPC_FOPDT_emulation_result.pkl',mode='wb')
dump([SP1, SP2, y1_mpc, y2_mpc, y1_lstm, y2_lstm, y1_trans, y2_trans,
    u1_mpc, u2_mpc, u1_lstm, u2_lstm, u1_trans, u2_trans], file)
file.close()


data = pd.read_pickle('MPC_FOPDT_emulation_result.pkl')
SP1 = data[0]
SP2 = data[1]
y1_mpc = data[2]
y2_mpc = data[3]
y1_lstm = data[4]
y2_lstm = data[5]
y1_trans = data[6]
y2_trans = data[7]
u1_mpc = data[8]
u2_mpc = data[9]
u1_lstm = data[10]
u2_lstm = data[11]
u1_trans = data[12]
u2_trans = data[13]



#%% mpl.style.use('default')
plt.figure(0, figsize=(8,6))
plt.subplot(2,2,1)
plt.plot(SP1[window:], 'tab:red', lw=2, drawstyle='steps')
plt.plot(y1_mpc, 'k', ls="-", lw=1.5)
plt.plot(y1_lstm[window+1:], 'tab:orange', ls="--")
plt.plot(y1_trans[window+1:], 'tab:blue', ls="--")
plt.legend(['SP', 'MPC', 'LSTM', 'Transformer'], loc=4, fontsize=10)
plt.ylabel('y1')

plt.subplot(2,2,2)
plt.plot(SP2[window:], 'tab:red', lw=2, drawstyle='steps')
plt.plot(y2_mpc, 'k', ls="-", lw=1.5)
plt.plot(y2_lstm[window+1:], 'tab:orange', ls="--")
plt.plot(y2_trans[window+1:], 'tab:blue', ls="--")
plt.legend(['SP', 'MPC',  'LSTM', 'Transformer'], loc=4, fontsize=10)
plt.ylabel('y2')

plt.subplot(2,2,3)
plt.plot(u1_mpc, 'k', drawstyle='steps')
plt.plot(u1_lstm[window+1:], 'tab:orange', ls="--", drawstyle='steps')
plt.plot(u1_trans[window+1:], 'tab:blue', ls="--", drawstyle='steps')
plt.legend(['MPC','LSTM', 'Transformer'], loc=4, fontsize=10)
plt.xlabel('Time (minute)')
plt.ylabel('u1')

plt.subplot(2,2,4)
plt.plot(u2_mpc, 'k', drawstyle='steps')
plt.plot(u2_lstm[window+1:], 'tab:orange', ls="--", drawstyle='steps')
plt.plot(u2_trans[window+1:], 'tab:blue', ls="--", drawstyle='steps')
plt.legend(['MPC','LSTM','Transformer'], loc=4, fontsize=10)
plt.xlabel('Time (minute)')
plt.ylabel('u2')

# plt.savefig('MPC_FOPDT_emulation_result.eps', format='eps')
plt.show()

#%%



# Plotly

# fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
# fig.add_trace(go.Scatter(y=SP1[window:],name="$SP1$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=SP2[window:],name="$SP2$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y1_mpc,name="$y1_{MPC}$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y2_mpc,name="$y2_{MPC}$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y1_lstm[window+1:],name="$y1_{LSTM}$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y2_lstm[window+1:],name="$y2_{LSTM}$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y1_trans[window+1:],name="$y1_{Transformer}$"), row=1, col=1)
# fig.add_trace(go.Scatter(y=y2_trans[window+1:],name="$y2_{Transformer}$"), row=1, col=1)

# fig.add_trace(go.Scatter(y=u1_mpc[window+1:],name="$u1_{MPC}$"), row=2, col=1)
# fig.add_trace(go.Scatter(y=u2_mpc[window+1:],name="$u2_{MPC}$"), row=2, col=1)
# fig.add_trace(go.Scatter(y=u1_lstm[window+1:],name="$u1_{LSTM}$"), row=2, col=1)
# fig.add_trace(go.Scatter(y=u2_lstm[window+1:],name="$u2_{LSTM}$"), row=2, col=1)
# fig.add_trace(go.Scatter(y=u1_trans[window+1:],name="$u1_{Transformer}$"), row=2, col=1)
# fig.add_trace(go.Scatter(y=u2_trans[window+1:],name="$u2_{Transformer}$"), row=2, col=1)

# fig.update_layout(
#     template="plotly_white",
#     font_family="Times New Roman",
#     font_size = 20
#     )

# fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True, row=1 , col=1)
# fig.update_xaxes(title_text='Time',showline=True, linewidth=2, linecolor='black', mirror=True, row=2, col=1)
# fig.update_yaxes(title_text='y', showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=1)
# fig.update_yaxes(title_text='u', showline=True, linewidth=2, linecolor='black', mirror=True, row=2, col=1)
# fig.show()

