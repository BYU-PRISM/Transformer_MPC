from functions.roaster_process import ProcessModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO

path = 'data/'   


#%% Load CSV files

# Load Thermodyanamic Data
data = pd.read_csv(path + 'ThermoData.csv', index_col = 0)

# Load Stoichiometry data
Stoi = pd.read_csv(path + 'Stoichiometry_Test4.csv', index_col = 0)
Stoi = Stoi.T
Stoi.S[2:] = Stoi.S[2:]/Stoi.S[1]
Stoi.TCM[2:] = Stoi.TCM[2:]/Stoi.TCM[1]
Stoi.FeS2[2:] = Stoi.FeS2[2:]/Stoi.FeS2[1]
Stoi.Fe087S[2:] = Stoi.Fe087S[2:]/Stoi.Fe087S[1]
Stoi.SO2[2:] = Stoi.SO2[2:]/Stoi.SO2[1]


Stoi = Stoi.T

# Read FV.NEWVAL from csv file for simulation (for IMODE = 1, 3 or 4)
FVs = pd.read_csv(path + "FVs.csv", index_col=0)

# from smt.sampling_methods import LHS
# from smt.sampling_methods import Random


#%% Sampling

# ninput = 6 # number of inputs
# xlimits = np.array([[80, 100], [9, 13], [7000, 8000], [0.8, 0.9], [0.6, 0.7], [17, 18]]) # time step for input1, 2, 3 and input1, 2, and 3

# # Latin Hypercube Sampling
# # sampling = LHS(xlimits=xlimits)
# # x = sampling(num[i])


# # Random Sampling
# sampling = Random(xlimits=xlimits)
# x = sampling(num[i])


# # Generating the time points for step changes 
# t_change = np.random.randint(10, 30, [num[i],ninput])
# for k in range(1,np.shape(t_change)[0]):
#     t_change[k] = t_change[k]+t_change[k-1]

# print(x.shape)

p = ProcessModel(data,Stoi,FVs)


[T1, T2] = p.run()

# %% plotting
t_minute = p.time/60

plt.figure(1)
plt.title('Feed rate')
plt.subplot(3,1,1)
plt.plot(t_minute,p.data_input["Ore_amps"], color='tab:blue', linestyle='-', linewidth=3,label='Ore_amps')
plt.ylabel('Amps')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(t_minute,p.data_input["Sulfur_tph"], color='tab:blue', linestyle='-', linewidth=3,label='Sulfur_tph')
plt.ylabel('t/h')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,3)
plt.plot(t_minute,p.data_input["O2_scfm"], color='tab:blue', linestyle='-', linewidth=3,label='O2_scfm')
plt.ylabel('SCFM[kg/h]')
plt.xlabel('time[min]')
plt.legend(loc='best') 

plt.figure(2)
plt.title('Feed Ore Composition')
plt.subplot(3,1,1)
plt.plot(t_minute,p.data_input["Carbon_in"], color='tab:orange', linestyle='-', linewidth=3,label='Carbon_in (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(t_minute,p.data_input["Sulf_in"], color='tab:orange', linestyle='-', linewidth=3,label='Sulf_in (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(t_minute,p.data_input["CO3_in"], color='tab:orange', linestyle='-', linewidth=3,label='CO3_in (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')  

# plt.figure(3)
# plt.subplot(4,1,1)
# plt.plot(t_minute,data_input["level1"], color='tab:orange', linestyle='-', linewidth=3,label='level1')
# plt.ylabel('%')
# plt.xlabel('time[min]')
# plt.legend(loc='best') 
# plt.subplot(4,1,2)
# plt.plot(t_minute,tau1_m, color='tab:green', linestyle='-', linewidth=3,label='Residence time 1')
# plt.ylabel('%')
# plt.xlabel('time[min]')
# plt.legend(loc='best') 
# plt.subplot(4,1,3)
# plt.plot(t_minute,data_input["level2"], color='tab:orange', linestyle='-', linewidth=3,label='level2')
# plt.ylabel('%')
# plt.xlabel('time[min]')
# plt.legend(loc='best') 
# plt.subplot(4,1,4)
# plt.plot(t_minute,tau2_m, color='tab:green', linestyle='-', linewidth=3,label='Residence time 1')
# plt.ylabel('%')
# plt.xlabel('time[min]')
# plt.legend(loc='best')  

plt.figure(4)
plt.title('Off Gas Composition')
plt.subplot(3,1,1)
plt.plot(t_minute,p.wp_og1["O2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_O2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(t_minute,p.wp_og1["CO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_CO2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(t_minute,p.wp_og1["SO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_SO2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     

plt.figure(5)
plt.title('Calcine Composition')
plt.subplot(3,1,1)
plt.plot(t_minute,p.wp_calcine2["TCM"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_TCM (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(t_minute,p.wp_calcine2["FeS2"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_Sulfur (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(t_minute,p.wp_calcine2["CaCO3"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_CO3 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')  

plt.figure(6)
plt.title("Reactor Temperature")
plt.subplot(2,1,1)
plt.plot(t_minute,T1, color='tab:red', linestyle='-', linewidth=3,label='T1')
plt.ylabel('F')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(2,1,2)
plt.plot(t_minute,T2, color='tab:red', linestyle='-', linewidth=3,label='T2')
plt.ylabel('F')
plt.xlabel('time[min]')
plt.legend(loc='best')     

plt.show()