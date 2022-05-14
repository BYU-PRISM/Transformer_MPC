from functions.roaster_process import ProcessModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO

path = 'data/'   


# Load data from other code to compare (just for test)
testdata = pd.read_pickle('Roaster_data_training_random10.pkl')

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

from smt.sampling_methods import LHS
from smt.sampling_methods import Random


#%% Sampling
# num = [50, 70, 90, 110] # number of steps (or, number of samples)
num = [150]
for i in range(np.size(num)):
    print(i)
  
    
    #%% Sampling
    
    ninput = 6 # number of inputs
    xlimits = np.array([[80, 100], [9, 13], [7000, 8000], [0.8, 0.9], [0.6, 0.7], [17, 18]]) # time step for input1, 2, 3 and input1, 2, and 3
    
    # Latin Hypercube Sampling
    sampling = LHS(xlimits=xlimits)
    x = sampling(num[i])
    
    
    # Random Sampling
    # sampling = Random(xlimits=xlimits)
    # x = sampling(num[i])
    
    
    # Generating the time points for step changes 
    t_change = np.random.randint(10, 30, [num[i],ninput])
    for k in range(1,np.shape(t_change)[0]):
        t_change[k] = t_change[k]+t_change[k-1]
    
    print(x.shape)
    
    # nstep = np.max(t_change)+30
    nstep = 237
    
      
    #%% input data
    
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
    

    

    # # Creating time points to step changes
    # for j in range(num[i]):
    #     data_input["Ore_amps"][t_change[j,0]:] = x[j,0]
    #     data_input["Sulfur_tph"][t_change[j,1]:] = x[j,1]
    #     data_input["O2_scfm"][t_change[j,2]:] = x[j,2]
    #     data_input["Carbon_in"][t_change[j,3]:] = x[j,3]
    #     data_input["Sulf_in"][t_change[j,4]:] = x[j,4]
    #     data_input["CO3_in"][t_change[j,5]:] = x[j,5]

    data_input["Ore_amps"] = testdata["Ore_amps"]
    data_input["Sulfur_tph"] = testdata["Sulfur_tph"]
    data_input["O2_scfm"] = testdata["O2_scfm"]
    data_input["Carbon_in"] = testdata["Carbon_in"]
    data_input["Sulf_in"] = testdata["Sulf_in"]
    data_input["CO3_in"] = testdata["CO3_in"]






df = pd.DataFrame(data_input)
input = df.to_numpy()
# input = input[:,:6]

ns = 60  # Simulation Length
t = np.linspace(0, ns, ns + 1)
delta_t = 10


# nu = 6
# ny = 8
p = ProcessModel(data,Stoi,FVs, delta_t)

for i in range(1, ns):

    # run process model
    p.run(input[i])
    print(i)

result = p.get_result()

# result["O2"]


# %% plotting
# t_minute = p.time/60

plt.figure(1)
plt.title('Feed rate')
plt.subplot(3,1,1)
plt.plot(data_input["Ore_amps"][0:ns], color='tab:blue', linestyle='-', linewidth=3,label='Ore_amps')
plt.ylabel('Amps')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(data_input["Sulfur_tph"][0:ns], color='tab:blue', linestyle='-', linewidth=3,label='Sulfur_tph')
plt.ylabel('t/h')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,3)
plt.plot(data_input["O2_scfm"][0:ns], color='tab:blue', linestyle='-', linewidth=3,label='O2_scfm')
plt.ylabel('SCFM[kg/h]')
plt.xlabel('time[min]')
plt.legend(loc='best') 

plt.figure(2)
plt.title('Feed Ore Composition')
plt.subplot(3,1,1)
plt.plot(data_input["Carbon_in"][0:ns], color='tab:orange', linestyle='-', linewidth=3,label='Carbon_in (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(data_input["Sulf_in"][0:ns], color='tab:orange', linestyle='-', linewidth=3,label='Sulf_in (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(data_input["CO3_in"][0:ns], color='tab:orange', linestyle='-', linewidth=3,label='CO3_in (wt%)')
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
plt.plot(result["O2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_O2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(result["CO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_CO2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(result["SO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_SO2 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     

plt.figure(5)
plt.title('Calcine Composition')
plt.subplot(3,1,1)
plt.plot(result["TCM"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_TCM (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(3,1,2)
plt.plot(result["FeS2"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_Sulfur (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')     
plt.subplot(3,1,3)
plt.plot(result["CaCO3"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_CO3 (wt%)')
plt.ylabel('wt%')
plt.xlabel('time[min]')
plt.legend(loc='best')  

plt.figure(6)
plt.title("Reactor Temperature")
plt.subplot(2,1,1)
plt.plot(result["T1"], color='tab:red', linestyle='-', linewidth=3,label='T1')
plt.ylabel('F')
plt.xlabel('time[min]')
plt.legend(loc='best') 
plt.subplot(2,1,2)
plt.plot(result["T2"], color='tab:red', linestyle='-', linewidth=3,label='T2')
plt.ylabel('F')
plt.xlabel('time[min]')
plt.legend(loc='best')     

plt.show()



print("break")

data1 = pd.DataFrame(data_input)
data2 = pd.DataFrame(result)
data_all = pd.concat((data1, data2), axis=1)

data_all.to_pickle('Roaster_data_training_random_test.pkl')