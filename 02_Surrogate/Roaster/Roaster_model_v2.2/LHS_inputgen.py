import Gekko_Roaster_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from smt.sampling_methods import LHS
from smt.sampling_methods import Random


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
    nstep = 3000
    
      
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
    
    # Creating time points to step changes
    for j in range(num[i]):
        data_input["Ore_amps"][t_change[j,0]:] = x[j,0]
        data_input["Sulfur_tph"][t_change[j,1]:] = x[j,1]
        data_input["O2_scfm"][t_change[j,2]:] = x[j,2]
        data_input["Carbon_in"][t_change[j,3]:] = x[j,3]
        data_input["Sulf_in"][t_change[j,4]:] = x[j,4]
        data_input["CO3_in"][t_change[j,5]:] = x[j,5]
               
     # unit conversions to tons/hr
    data_tph = {
        "Ore_in": data_input["Ore_amps"] * 1.675786 + 208.270,
        "Ore_in_HI": data_input["Ore_in_HI"] * 1.675786 + 208.270,
        "Ore_in_LO": data_input["Ore_in_LO"] * 1.675786 + 208.270,
        "Sulfur_in": data_input["Sulfur_tph"],
        "Sul_in_HI": data_input["Sul_in_HI"],
        "Sul_in_LO": data_input["Sul_in_LO"],
        "O2_in": data_input["O2_scfm"] * 0.002421,
        "O2_in_HI": data_input["O2_in_HI"] * 0.002421,
        "O2_in_LO": data_input["O2_in_LO"] * 0.002421,
        "Gold_in": 0
          }


    data_tph["Gold_in"] = data_input["Gold_in"] * data_tph["Ore_in"]* 2.835e-5 # unit conversion from 'oz/ton' to 'ton/hr'
    
    key_csv_wtpercent_in = {"Carbon_in", "CO3_in", "Sulf_in"}
    key_csv_wtpercent_out = {"O2_out", "CO2_out", "SO2_out", "CO3_out", "TCM_out", "Sulf_out"}
    
    # wt% to ton/h
    for key in key_csv_wtpercent_in:
        data_tph[key] = data_input[key] * data_tph["Ore_in"] * 1e-2
    
    class input():
        Sulf_in = []
        CO3_in = []
        Carbon_in = []
        Gold_in = []
        Ore_in = []
        Sulfur_in = []
        O2_in = []
    
    # Arrays for storing output data from gekko
    O2 = np.zeros(nstep)
    CO2 = np.zeros(nstep)
    SO2 = np.zeros(nstep)
    TCM = np.zeros(nstep)
    FeS2 = np.zeros(nstep)
    CaCO3 = np.zeros(nstep)
    T_1 = np.zeros(nstep)
    T_2 = np.zeros(nstep)



    class pred():
        Ore_amps = []
        Sulfur_tph = []
        O2_scfm = []
        Carbon_in = []
        Sulf_in = []
        CO3_in = []
        O2 = []
        CO2 = []
        SO2 = []
        TCM = []
        FeS2 = []
        CaCO3 = []
        T_1 = []
        T_2 = []
       
    # # Arrays for storing prediction data from lstm
    # O2_pred = np.zeros(nstep)
    # CO2_pred = np.zeros(nstep)
    # SO2_pred = np.zeros(nstep)
    # TCM_pred = np.zeros(nstep)
    # FeS2_pred = np.zeros(nstep)
    # CaCO3_pred = np.zeros(nstep)
    # T_1_pred = np.zeros(nstep)
    # T_2_pred = np.zeros(nstep)
    
    #%% Running process model (Gekko)
    # Create plot
    nplot = 4
    # for i in range(nplot):
    #     plt.figure(i)
    #     plt.ion()
    # plt.show()
        
    
    input.Sulf_in = data_tph["Sulf_in"]
    input.CO3_in = data_tph["CO3_in"]
    input.Carbon_in = data_tph["Carbon_in"]
    input.Gold_in = data_tph["Gold_in"]
    input.Ore_in = data_tph["Ore_in"]
    input.Sulfur_in = data_tph["Sulfur_in"]
    input.O2_in = data_tph["O2_in"]
    
    output = Gekko_Roaster_model.process(input)
    
    
    O2 = output.O2
    CO2 = output.CO2
    SO2 = output.SO2
    TCM = output.TCM
    FeS2 = output.FeS2
    CaCO3 = output.CaCO3
    T_1 = output.T_1
    T_2 = output.T_2  

 
    
    #%% save data to csv
    df = pd.DataFrame()
    df['Ore_amps'] = data_input["Ore_amps"]
    df['Sulfur_tph'] = data_input["Sulfur_tph"]
    df['O2_scfm'] = data_input["O2_scfm"]
    df['Carbon_in'] = data_input["Carbon_in"]
    df['Sulf_in'] = data_input["Sulf_in"]
    df['CO3_in'] = data_input["CO3_in"]
    df['O2'] = O2
    df['CO2'] = CO2
    df['SO2'] = SO2
    df['TCM'] = TCM
    df['FeS2'] = FeS2
    df['CaCO3'] = CaCO3
    df['T_1'] = T_1
    df['T_2'] = T_2
    
    df.to_csv('Roaster_data_random'+str(num[i])+'.csv')