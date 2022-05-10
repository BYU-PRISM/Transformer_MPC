from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO
from smt.sampling_methods import LHS
from smt.sampling_methods import Random

path = 'data/'

num = [10] # number of steps (or, number of samples)
for i in range(np.size(num)):
    print(i)
    
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
    
    #%% Sampling
    
    ninput = 6 # number of inputs
    xlimits = np.array([[80, 100], [9, 13], [7000, 8000], [0.8, 0.9], [0.6, 0.7], [17, 18]]) # time step for input1, 2, 3 and input1, 2, and 3
    
    # Latin Hypercube Sampling
    # sampling = LHS(xlimits=xlimits)
    # x = sampling(num[i])
    
    
    # Random Sampling
    sampling = Random(xlimits=xlimits)
    x = sampling(num[i])
    
    
    # Generating the time points for step changes 
    t_change = np.random.randint(10, 30, [num[i],ninput])
    for k in range(1,np.shape(t_change)[0]):
        t_change[k] = t_change[k]+t_change[k-1]
    
    print(x.shape)
    
    nstep = np.max(t_change)+30
    
    #%% simulation settings
    
    # Simulation Time
    # tsim = 11000* 60 # sec
    tsim = nstep * 10 * 60 # [sec] = nstep * [min/step] * [sec/min] 
    
    # Time step
    tstep = 10 * 60 # sec
    
    # size of array (number of steps)
    # nstep = int(tsim/tstep+1)
    
    # Sulphur Particle Radius
    R_Sulphur = 3e-3 # [m] (0.5 mm) 
        
    # Ore Particle Radius
    R_Ore = 75e-06/2 #[m] (75 micrometer)
    
    # Residence time (sec)
    tau1_ss = 20 * 60
    tau2_ss = 22 * 60
    
    # level
    level1_ss = 140
    level2_ss = 30
    
    # Steady State Bed Temperature  (F to K)
    T_ss1 = (1002 - 32) * 5/9 + 273.15  # [K]
    T_ss2 = (1030 - 32) * 5/9 + 273.15  # [K]
    
    
    #%% Load Validation Data
    # data_csv = pd.read_csv('data_validation3.csv', index_col = 0)
    
    # nsample = np.size(data_csv,0)
    
    # Data input in unit wt%, Fafahrenheit, amps (Feed_amps), scfm (O2_in), ton/h(Feed_Sulfur), oz/ton (Gold)
    key_csv = {"Ore_amps", "Sulfur_tph", "O2_scfm",
               "Carbon_in", "Sulf_in",	"CO3_in", "Gold_in",
               "O2_out", "CO2_out",	"SO2_out", "CO3_out", "TCM_out", "Sulf_out", "Gold_out",
               "T1", "T2"}
    
    #%% input data
    
    data_input = {
            "Ore_amps" : np.ones(nstep)*90,
            "Sulfur_tph": np.ones(nstep)*11,
            "O2_scfm": np.ones(nstep)*7400,
            "Carbon_in": np.ones(nstep)*0.87,
            "Sulf_in": np.ones(nstep)*0.64,
            "CO3_in": np.ones(nstep)*17.92,
            "Gold_in":np.ones(nstep)*0.16,
            "level1": np.ones(nstep)*level1_ss,
            "level2": np.ones(nstep)*level2_ss,
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
               
       
    # plt.figure(0)
    # plt.subplot(3,1,1)
    # plt.plot(data_input["Ore_amps"])
    # plt.subplot(3,1,2)
    # plt.plot(data_input["Sulfur_tph"])
    # plt.subplot(3,1,3)
    # plt.plot(data_input["O2_scfm"])
    
    # data_input["Sulfur_tph"][30:] = 12
    # data_input["Sulfur_tph"][60:] = 13
    # data_input["Sulfur_tph"][90:] = 14
    # data_input["Sulfur_tph"][120:] = 15
    # data_input["Sulfur_tph"][150:] = 16
    # data_input["Sulfur_tph"][180:] = 17
    # data_input["Sulfur_tph"][210:] = 18
    
    
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
        
    
    #%% GEKKO
    m = GEKKO(remote=True)
    #m.time = [0,tstep]           # for animated version
    m.time = np.linspace(0, tsim, nstep)
    
    # TPH
    S2 = m.MV(0)
    CO3 = m.MV(0)
    TCM = m.MV(0)
    
    Ore_in = m.MV(0)
    Sul_in = m.MV(0)
    O2_in = m.MV(0)
    Gold_in = m.MV(0)
    
    level1 = m.MV(level1_ss, ub=150, lb=130)
    level2 = m.MV(level2_ss, ub=45, lb=15)
    
    a1 = 0.4
    b1 = -36
    a2 = 0.2
    b2 = 16
    
    tau1 = m.Intermediate((a1*level1+b1)*60)
    tau2 = m.Intermediate((a2*level2+b2)*60)
    
    SiO2 = m.Intermediate(Ore_in - (S2+CO3+TCM)) 
    
    O2_bias = m.FV(0)
    air = m.FV(0)
    calcine_bias = m.FV(0)
    
    T_Ore0 = 160 # F
    T_Sulphur0 = 68 # F
    T_O20 = 75 # F
    
    #%% Unit conversion
    
    # F to K
    T_Ore0 =(T_Ore0 -32) * 5/9 + 273.15 #[K]
    T_Sulphur0 = (T_Sulphur0 -32) * 5/9 + 273.15 #[K]
    T_O20 = (T_O20 -32) * 5/9 + 273.15 #[K]
    
    #%%
    # Separate list of keys for using in 'For' loop 
    keys_Solid = ['SiO2', 'FeS2', 'Fe087S', 'CaCO3', 'TCM', 'Au', 'CaSO4', 'Fe2O3', 'S']
    keys_Gas = ['CO2','O2','SO2']
    
    keys = keys_Solid + keys_Gas
    
    # Feed mass Flowrate (Ton/h)
    w0 = {
        'SiO2': m.Intermediate(SiO2),
        'FeS2': m.Intermediate(S2),
        'Fe087S': 0,
        'CaCO3': m.Intermediate(CO3),
        'TCM': m.Intermediate(TCM),
        'Au': m.Intermediate(Gold_in),
        'CaSO4' : 0,
        'Fe2O3': 0,
        'S': m.Intermediate(Sul_in),
        'CO2': 0,
        'O2': m.Intermediate(O2_in + O2_bias),
        'SO2': 0,
    }
    
    # TPH to moles/sec
    y0 = {}
    for key in w0.keys():
        y0[key] = m.Intermediate(w0[key] / data[key].MW * 1e6/3600)
    
    yOut1 = {}  # Number of Moles of 1st Stage Outlet
    yOut2 = {}  # Number of Moles of  1st Stage Outlet
    yog1 = {}   # Number of Moles of Offgas from 1st stage
    yog2 = {}   # Number of Moles of Offgas from 1st stage
    ycalcine1 = {} # Number of Moles of Calcine from 1st stage 
    ycalcine2 = {} # Number of Moles of Calcine from 1st stage
    yIn1 = {}   # Number of Moles of 1st Stage Inlet   
    yIn2 = {}   # Number of Moles of 2nd Stage Inlet
    r1 = {}     # Reaction constant for 1st Stage
    r2 = {}     # Reaction constant for 2nd Stage
    
    for key in keys_Solid:
        yOut1[key] = m.CV(value=0)
        yOut2[key] = m.CV(value=0)
        yog1[key] = m.Intermediate(0)
        yog2[key] = m.Intermediate(0)
        ycalcine1[key] = m.Intermediate(yOut1[key])
        ycalcine2[key] = m.Intermediate(yOut2[key])
        yIn1[key] = m.Intermediate(y0[key])
        yIn2[key] = m.Intermediate(ycalcine1[key])
        
    for key in keys_Gas:
        yOut1[key] = m.CV(value=0)
        yOut2[key] = m.CV(value=0)
        yog1[key] = m.Intermediate(yOut1[key])
        yog2[key] = m.Intermediate(yOut2[key])
        ycalcine1[key] = m.Intermediate(0)
        ycalcine2[key] = m.Intermediate(0)
        yIn1[key] = m.Intermediate(yog2[key])
        yIn2[key] = m.Intermediate(y0[key])
        
        
    #%% Convert Number of Moles to Mass (T/h)
    
    TCM_bias = m.FV(0, ub=2, lb=-2)    
    CO3_bias = m.FV(0, ub=50, lb=-50)
    Sulf_bias = m.FV(0, ub=10, lb=-10)
        
    w0 = {}
    wIn1 = {}
    wOut1 = {}
    wog1 = {}
    wcalcine1 = {}
    wIn2 = {}
    wOut2 = {}
    wog2 = {}
    wcalcine2 = {}
    wOut = {}
    for key in keys:
        w0[key] = m.Intermediate(y0[key] * data[key].MW * 3600/1e6)
        
        wIn1[key] = m.Intermediate(yIn1[key] * data[key].MW * 3600/1e6)
        wOut1[key] =  m.Intermediate(yOut1[key] * data[key].MW * 3600/1e6)
        wog1[key] =  m.Intermediate(yog1[key] * data[key].MW * 3600/1e6)
        wcalcine1[key] =  m.Intermediate(ycalcine1[key] * data[key].MW * 3600/1e6)
        
        wIn2[key] =  m.Intermediate(yIn2[key] * data[key].MW * 3600/1e6)
        wOut2[key] =  m.Intermediate(yOut2[key] * data[key].MW * 3600/1e6)
        wog2[key] =  m.Intermediate(yog2[key] * data[key].MW * 3600/1e6)
        wcalcine2[key] =  m.Intermediate(ycalcine2[key] * data[key].MW * 3600/1e6)
        
        wOut[key] = m.Intermediate(wcalcine2[key] + wog1[key])
        
    wcalcine2["TCM"] = m.Intermediate(wcalcine2["TCM"]+TCM_bias)
    wcalcine2["CaCO3"] = m.Intermediate(wcalcine2["CaCO3"]+CO3_bias)
    wcalcine2["FeS2"] = m.Intermediate(wcalcine2["FeS2"]+Sulf_bias)
    
    w0_flowrate = m.Intermediate(w0["SiO2"] + w0["FeS2"] + w0["Fe087S"]\
                                 + w0["CaCO3"] + w0["TCM"] + w0["Au"]\
                                 + w0["CaSO4"] + w0["Fe2O3"] + w0["S"]\
                                 + w0["CO2"] + w0["O2"] + w0["SO2"])
          
    wIn1_flowrate = m.Intermediate(wIn1["SiO2"] + wIn1["FeS2"] + wIn1["Fe087S"]\
                                        + wIn1["CaCO3"] + wIn1["TCM"] + wIn1["Au"]\
                                        + wIn1["CaSO4"] + wIn1["Fe2O3"] + wIn1["S"]\
                                        + wIn1["CO2"] + wIn1["O2"] + wIn1["SO2"])
    
    wcalcine1_flowrate = m.Intermediate(wcalcine1["SiO2"] + wcalcine1["FeS2"] + wcalcine1["Fe087S"]\
                                        + wcalcine1["CaCO3"] + wcalcine1["TCM"] + wcalcine1["Au"]\
                                        + wcalcine1["CaSO4"] + wcalcine1["Fe2O3"] + wcalcine1["S"]\
                                        + wcalcine1["CO2"] + wcalcine1["O2"] + wcalcine1["SO2"])
    
    wcalcine2_flowrate = m.Intermediate(wcalcine2["SiO2"] + wcalcine2["FeS2"] + wcalcine2["Fe087S"]\
                                        + wcalcine2["CaCO3"] + wcalcine2["TCM"] + wcalcine2["Au"]\
                                        + wcalcine2["CaSO4"] + wcalcine2["Fe2O3"] + wcalcine2["S"]\
                                        + wcalcine2["CO2"] + wcalcine2["O2"] + wcalcine2["SO2"])
    wog1_flowrate = m.Intermediate(wog1["SiO2"] + wog1["FeS2"] + wog1["Fe087S"]\
                                        + wog1["CaCO3"] + wog1["TCM"] + wog1["Au"]\
                                        + wog1["CaSO4"] + wog1["Fe2O3"] + wog1["S"]\
                                        + wog1["CO2"] + wog1["O2"] + wog1["SO2"] + air)
    wog2_flowrate = m.Intermediate(wog2["SiO2"] + wog2["FeS2"] + wog2["Fe087S"]\
                                        + wog2["CaCO3"] + wog2["TCM"] + wog2["Au"]\
                                        + wog2["CaSO4"] + wog2["Fe2O3"] + wog2["S"]\
                                        + wog2["CO2"] + wog2["O2"] + wog2["SO2"])
    
    wOut_flowrate = m.Intermediate(wOut["SiO2"] + wOut["FeS2"] + wOut["Fe087S"]\
                                        + wOut["CaCO3"] + wOut["TCM"] + wOut["Au"]\
                                        + wOut["CaSO4"] + wOut["Fe2O3"] + wOut["S"]\
                                        + wOut["CO2"] + wOut["O2"] + wOut["SO2"])
    
    
    # Weight Percent Calculation
    wp_og1 = {}
    wp_calcine1 = {}
    wp_og2 = {}
    wp_calcine2 = {}
        
    for key in keys:  
        wp_og1[key] = m.CV(0)
        wp_calcine1[key] = m.CV(0)
        wp_og2[key] = m.CV(0)
        wp_calcine2[key] = m.CV(0)
    
    for key in keys:  
        m.Equation(wp_og1[key] ==  (wog1[key] / wog1_flowrate) * 100) 
        m.Equation(wp_calcine1[key] ==  (wcalcine1[key] / wcalcine1_flowrate)*100)
        m.Equation(wp_og2[key] ==  (wog2[key] / wog2_flowrate) * 100)
        m.Equation(wp_calcine2[key] ==  (wcalcine2[key] / wcalcine2_flowrate) * 100) 
    
    #%% Rate Constant for Coal (m/s)
    T_reactor1 = m.CV(value=T_ss1)
    T_reactor2 = m.CV(value=T_ss2)
    P_reactor = 10e5 # Pa
    
    # Pre-exponential factor (m/s)
    k0_S = 2e20
    k0_Ore = 1.1e20
    k0_Sulphate = 1.1e20
    k0_TCM = 6.0e20
    k0_SO2 = 1.0e20
    
    # E - Activation energy in the Arrhenius Equation (J/mol)
    # R - Universal Gas Constant = 8.31451 J/mol-K
    EoverR_S = 184000/8.31451
    EoverR_Ore = 184000/8.31451
    EoverR_Sulphate = 184000/8.31451
    EoverR_TCM = 190000/8.31451
    EoverR_SO2 = 184000/8.31451
    
    # Chapman Enskog for D0
    p = 1
    sigma_A = 3.433
    sigma_B = 3.433
    sigma_avg = 1/2*(sigma_A + sigma_B)
    
    eps_over_K_A  = 113
    eps_over_K_B = 113
    eps_over_K_avg = np.sqrt(eps_over_K_A * eps_over_K_B)
    
    T_star1 = m.Intermediate(T_reactor1/eps_over_K_avg)
    T_star2 = m.Intermediate(T_reactor2/eps_over_K_avg)
    
    mw_A = 32
    mw_B = 32
    
    ohm1 = m.Intermediate(1.06036/T_star1**0.1561 + 0.19300/m.exp(0.47635* T_star1) + 1.03587/m.exp(1.52996*T_star1) + 1.76474 / m.exp(3.89411*T_star1))
    ohm2 = m.Intermediate(1.06036/T_star2**0.1561 + 0.19300/m.exp(0.47635* T_star2) + 1.03587/m.exp(1.52996*T_star2) + 1.76474 / m.exp(3.89411*T_star2))
    
    D01 = m.Intermediate(0.0018583 * m.sqrt(T_reactor1**3 * (1/mw_A + 1/mw_B))* 1/(p * sigma_avg**2 * ohm1)) # m2/s
    D02 = m.Intermediate(0.0018583 * m.sqrt(T_reactor2**3 * (1/mw_A + 1/mw_B))* 1/(p * sigma_avg**2 * ohm1)) # m2/s
    
    eps_p =m.Param(0.5) # Porosity
    
    De1 = m.Intermediate(D01*eps_p**0.02)
    De2 = m.Intermediate(D02*eps_p**0.02)
    
    alpha_S = m.FV(10000, lb=10, ub=300000)
    alpha_Ore = m.FV(40000)#, lb=35000, ub=45000)
    alpha_Sulphate = m.FV(100000, lb=10, ub=200000)
    alpha_TCM = m.FV(85000, lb=10, ub=150000)
    alpha_SO2 = m.FV(150000, lb=10, ub=200000)
    
    kS1_ash = m.Intermediate((R_Sulphur/(6*De1)) * alpha_S)
    kOre1_ash = m.Intermediate((R_Ore/(6*De1)) * alpha_Ore)
    kSulphate1_ash = m.Intermediate((R_Ore/(6*De1)) * alpha_Sulphate)
    kTCM1_ash = m.Intermediate((R_Ore/(6*De1)) * alpha_TCM)
    kSO21_ash = m.Intermediate((R_Ore/(6*De1)) * alpha_SO2)
    
    kS2_ash = m.Intermediate((R_Sulphur/(6*De2))*alpha_S)
    kOre2_ash = m.Intermediate((R_Ore/(6*De2))*alpha_Ore)
    kSulphate2_ash = m.Intermediate((R_Ore/(6*De2))*alpha_Sulphate)
    kTCM2_ash = m.Intermediate((R_Ore/(6*De2))*alpha_TCM)
    kSO22_ash = m.Intermediate((R_Ore/(6*De2))*alpha_SO2)
    
    kS1_reaction = m.Intermediate(k0_S*m.sqrt(T_reactor1)*m.exp(-EoverR_S/T_reactor1))
    kOre1_reaction = m.Intermediate(k0_Ore*m.sqrt(T_reactor1)*m.exp(-EoverR_Ore/T_reactor1))
    kSulphate1_reaction = m.Intermediate(k0_Sulphate*m.sqrt(T_reactor1)*m.exp(-EoverR_Sulphate/T_reactor1))
    kTCM1_reaction = m.Intermediate(k0_TCM*m.sqrt(T_reactor1)*m.exp(-EoverR_TCM/T_reactor1))
    kSO21_reaction = m.Intermediate(k0_SO2*m.sqrt(T_reactor1)*m.exp(-EoverR_SO2/T_reactor1))
    
    kS2_reaction = m.Intermediate(k0_S*m.sqrt(T_reactor2)*m.exp(-EoverR_S/T_reactor2))
    kOre2_reaction = m.Intermediate(k0_Ore*m.sqrt(T_reactor2)*m.exp(-EoverR_Ore/T_reactor2))
    kSulphate2_reaction = m.Intermediate(k0_Sulphate*m.sqrt(T_reactor2)*m.exp(-EoverR_Sulphate/T_reactor2))
    kTCM2_reaction = m.Intermediate(k0_TCM*m.sqrt(T_reactor2)*m.exp(-EoverR_TCM/T_reactor2))
    kSO22_reaction = m.Intermediate(k0_SO2*m.sqrt(T_reactor2)*m.exp(-EoverR_SO2/T_reactor2))
    
    kS1 = m.Intermediate(1/(1/kS1_ash + 1/kS1_reaction))
    kOre1 = m.Intermediate(1/(1/kOre1_ash + 1/kOre1_reaction))
    kSulphate1 = m.Intermediate(1/(1/kSulphate1_ash + 1/kSulphate1_reaction))
    kTCM1 = m.Intermediate(1/(1/kTCM1_ash + 1/kTCM1_reaction))
    kSO21 = m.Intermediate(1/(1/kSO21_ash + 1/kSO21_reaction))
    
    kS2 = m.Intermediate(1/(1/kS2_ash + 1/kS2_reaction))
    kOre2 = m.Intermediate(1/(1/kOre2_ash + 1/kOre2_reaction))
    kSulphate2 = m.Intermediate(1/(1/kSulphate2_ash + 1/kSulphate2_reaction))
    kTCM2 = m.Intermediate(1/(1/kTCM2_ash + 1/kTCM2_reaction))
    kSO22 = m.Intermediate(1/(1/kSO22_ash + 1/kSO22_reaction))
    
    #%% Observed (Overall) rate constant
    k1 = {
        'S': kS1,
        'TCM': kTCM1,
        'FeS2': kOre1,
        'Fe087S': kSulphate1,
        'SO2': kSO21,
        'SiO2': 0, # Inert
        'Au': 0, # Inert
        }
    
    k2 = {
        'S': kS2,
        'TCM': kTCM2,
        'FeS2': kOre2,
        'Fe087S': kSulphate2,
        'SO2': kSO22,
        'SiO2': 0, # Inert
        'Au': 0, # Inert
    }
    
    #%% Rate equations
    
    # Specific surface
    eps = 0 
    a_Ore = 3*(1-0.99)/R_Ore
    a_Sulphur = 3*(1-0.99)/R_Sulphur
    V = np.pi * 5**2 * 2
    
    a = 0.002
    
    Asurf1 = {}
    Asurf2 = {}
    for key in k1.keys():
        Asurf1[key] = m.Intermediate(wOut1[key]*a)
        Asurf2[key] = m.Intermediate(wOut2[key]*a)
        
    P_O2 = 0.995e5 # Pascal
    C_O2_1 = m.Intermediate(P_O2/(8.314*T_reactor1)) # O2 Concentration
    C_O2_2 = m.Intermediate(P_O2/(8.314*T_reactor2)) # O2 Concentration
    
    
    r1["S"] = m.Intermediate(-(1 * kS1 * C_O2_1* Asurf1["S"]))
    r1["FeS2"] = m.Intermediate(-(1 * kOre1 * C_O2_1 * Asurf1["FeS2"]))
    r1["Fe087S"] = m.Intermediate(-(-1.142857 * kOre1 * C_O2_1 * Asurf1["FeS2"]) \
                                  - (1 * kSulphate1 * C_O2_1 * Asurf1["Fe087S"]))
    r1["TCM"] = m.Intermediate(-(1 * kTCM1 * C_O2_1 * Asurf1["TCM"]))
    r1["O2"] = m.Intermediate( -(1 * kS1 * C_O2_1 * Asurf1["S"]) \
                                -(1 * kTCM1 * C_O2_1 * Asurf1["TCM"]) \
                                 -(0.857143 * kOre1 * C_O2_1 * Asurf1["FeS2"]) \
                                  -(1.656250 * kSulphate1 * C_O2_1 * Asurf1["Fe087S"]) \
                                   -(0.5 * kSO21 * C_O2_1 * Asurf1["SO2"]))
    r1["CaCO3"] = m.Intermediate(-(1 * kSO21 * C_O2_1 * Asurf1["SO2"])) 
    r1["SO2"] = m.Intermediate(-(-1 * kS1 * C_O2_1 * Asurf1["S"]) \
                                 -(-0.857143 * kOre1 * C_O2_1 * Asurf1["FeS2"]) \
                                  -(-1 * kSulphate1 * C_O2_1 * Asurf1["Fe087S"]) \
                                   -(1 * kSO21 * C_O2_1 * Asurf1["SO2"])) 
    r1["CO2"] = m.Intermediate(-(-1 * kTCM1 * C_O2_1 * Asurf1["TCM"]) \
                                -(-1 * kSO21 * C_O2_1 * Asurf1["SO2"]))    
    r1["Fe2O3"] = m.Intermediate(-(-0.4375 * kSulphate1 * C_O2_1 * Asurf1["Fe087S"]))   
    r1["CaSO4"] = m.Intermediate(-(-1 * kSO21 * C_O2_1 * Asurf1["SO2"]))   
    r1["SiO2"] = m.Param(0)
    r1["Au"] = m.Param(0)    
    
    
    r2["S"] = m.Intermediate(-(1 * kS2 * C_O2_2* Asurf2["S"]))
    r2["FeS2"] = m.Intermediate(-(1 * kOre2 * C_O2_2 * Asurf2["FeS2"]))
    r2["Fe087S"] = m.Intermediate(-(-1.142857 * kOre2 * C_O2_2 * Asurf2["FeS2"]) \
                                  - (1 * kSulphate2 * C_O2_2 * Asurf2["Fe087S"]))
    r2["TCM"] = m.Intermediate(-(1 * kTCM2 * C_O2_2 * Asurf2["TCM"]))
    r2["O2"] = m.Intermediate( -(1 * kS2 * C_O2_2 * Asurf2["S"]) \
                                -(1 * kTCM2 * C_O2_2 * Asurf2["TCM"]) \
                                 -(0.857143 * kOre2 * C_O2_2 * Asurf2["FeS2"]) \
                                  -(1.656250 * kSulphate2 * C_O2_2 * Asurf2["Fe087S"]) \
                                   -(0.5 * kSO22 * C_O2_2 * Asurf2["SO2"]))
    r2["CaCO3"] = m.Intermediate(-(1 * kSO22 * C_O2_2 * Asurf2["SO2"])) 
    r2["SO2"] = m.Intermediate(-(-1 * kS2 * C_O2_2 * Asurf2["S"]) \
                                 -(-0.857143 * kOre2 * C_O2_2 * Asurf2["FeS2"]) \
                                  -(-1 * kSulphate2 * C_O2_2 * Asurf2["Fe087S"]) \
                                   -(1 * kSO22 * C_O2_2 * Asurf2["SO2"])) 
    r2["CO2"] = m.Intermediate(-(-1 * kTCM2 * C_O2_2 * Asurf2["TCM"]) \
                                -(-1 * kSO22 * C_O2_2 * Asurf2["SO2"]))    
    r2["Fe2O3"] = m.Intermediate(-(-0.4375 * kSulphate2 * C_O2_2 * Asurf2["Fe087S"]))   
    r2["CaSO4"] = m.Intermediate(-(-1 * kSO22 * C_O2_2 * Asurf2["SO2"]))   
    r2["SiO2"] = m.Param(0)
    r2["Au"] = m.Param(0)              
                               
    #%% Mass Balance
    for key in keys:
        m.Equation(yOut1[key].dt() == yIn1[key]/tau1 - yOut1[key]/tau1 + r1[key]) # MB for stage 1
        m.Equation(yOut2[key].dt() == yIn2[key]/tau2 - yOut2[key]/tau2 + r2[key]) # MB for stage 2
    
    #%% Heat In/Out
    
    # dH (J/mol) function
    T_st = 298.15
    
    def dH(a,b,c,d,T):
        dH = (a*T + (b/2000)*T**2 - (100000*c)/T + (d/3000000)*T**3) \
            - (a*T_st + (b/2000)*T_st**2 - (100000*c)/T_st + (d/3000000)*T_st**3)
        return dH
    
    keys_Ore = ['SiO2', 'FeS2', 'Fe087S', 'CaCO3', 'TCM', 'Au']
    keys_Sulphur = ['S']
    keys_calcine = keys_Solid
    keys_offgas = keys_Gas
    keys_feedgas = ['O2']
    
    # delH for Stage1 [J/mol]
    delH_In1_Ore = {}
    delH_In1_Sulphur = {}
    delH_In1_offgas = {}
    for key in keys_Ore:
        delH_In1_Ore[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_Ore0))
    for key in keys_Sulphur:
        delH_In1_Sulphur[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_Sulphur0))
    for key in keys_offgas:
        delH_In1_offgas[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor2))
    
    delH_Out1_calcine = {}
    delH_Out1_offgas = {}
    for key in keys_calcine:
        delH_Out1_calcine[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor1))
    for key in keys_offgas:    
        delH_Out1_offgas[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor1))
    
    # delH for Stage2 [J/mol]
    delH_In2_calcine = {}
    delH_In2_feedgas = {}
    for key in keys_calcine:
        delH_In2_calcine[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor1))
    for key in keys_feedgas:
        delH_In2_feedgas[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_O20))
    
    delH_Out2_calcine = {}
    delH_Out2_offgas = {}
    for key in keys_calcine:
        delH_Out2_calcine[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor2))
    for key in keys_offgas:
        delH_Out2_offgas[key] = m.Intermediate(dH(data[key].A, data[key].B, data[key].C, data[key].D, T_reactor2))
    
    # Grouping the Molar flowrate of each components for Heat flow calculation [J/s]
    yFeed_Ore = {}
    yFeed_Sulphur = {}
    ycalcine1_out = {}
    ycalcine2_out = {}
    yog1_out = {}
    yog2_out = {}
    yfeedgas = {}
    for key in keys_Ore:
        yFeed_Ore[key] = y0[key]
    for key in keys_Sulphur:
        yFeed_Sulphur[key] = y0[key]
    for key in keys_calcine:
        ycalcine1_out[key] = ycalcine1[key]
        ycalcine2_out[key] = ycalcine2[key]
    for key in keys_offgas:
        yog1_out[key] = yog1[key]
        yog2_out[key] = yog2[key]
    for key in keys_feedgas:
        yfeedgas[key] = y0[key]
    
    
    # Heat In and Out flow for whole components [J/s]
    Heat_In1_Ore = m.Intermediate(np.array(list(delH_In1_Ore.values())).dot(np.array(list(yFeed_Ore.values()))))
    Heat_In1_Sulphur = m.Intermediate(np.array(list(delH_In1_Sulphur.values())).dot(np.array(list(yFeed_Sulphur.values()))))
    Heat_In1_offgas = m.Intermediate(np.array(list(delH_In1_offgas.values())).dot(np.array(list(yog2_out.values()))))
    
    Heat_Out1_calcine = m.Intermediate(np.array(list(delH_Out1_calcine.values())).dot(np.array(list(ycalcine1_out.values()))))
    Heat_Out1_offgas = m.Intermediate(np.array(list(delH_Out1_offgas.values())).dot(np.array(list(yog1_out.values()))))
    
    Heat_In2_calcine = Heat_Out1_calcine
    Heat_In2_feedgas = m.Intermediate(np.array(list(delH_In2_feedgas.values())).dot(np.array(list(yfeedgas.values()))))
    
    Heat_Out2_calcine = m.Intermediate(np.array(list(delH_Out2_calcine.values())).dot(np.array(list(ycalcine2_out.values()))))
    Heat_Out2_offgas = m.Intermediate(np.array(list(delH_Out2_offgas.values())).dot(np.array(list(yog2_out.values()))))
    
    # Sum of the Heat flows of individual components 
    Heat_In1 = m.Intermediate(Heat_In1_Ore + Heat_In1_Sulphur + Heat_In1_offgas)
    Heat_Out1 = m.Intermediate(Heat_Out1_calcine + Heat_Out1_offgas)
    Heat_In2 = m.Intermediate(Heat_In2_calcine + Heat_In2_feedgas)
    Heat_Out2 = m.Intermediate(Heat_Out2_calcine + Heat_Out2_offgas)
    
    #%% Heat of Reaction (J/mol)
    Hrxn1 = {}
    rate1 = {}
    for key in k1.keys():
        Hrxn1[key] = -(Stoi.T.delH[2:].dot(Stoi.T[key][2:]))*1000
        rate1[key] = m.Intermediate(Stoi[key][key] * k1[key] * C_O2_1 * Asurf1[key] * tau1)
     
    Hgen1 = {}
    for key in k1.keys():
        Hgen1[key] = m.Intermediate(Hrxn1[key] * rate1[key])
    
    Heat_Gen1 = m.Intermediate(-1* np.array(list(Hrxn1.values())).dot(np.array(list(rate1.values()))))
    
    Hrxn2 = {}
    rate2 = {}
    for key in k2.keys():
        Hrxn2[key] = -(Stoi.T.delH[2:].dot(Stoi.T[key][2:]))*1000
        rate2[key] = m.Intermediate(Stoi[key][key] * k2[key] * C_O2_1 * Asurf2[key] * tau2)
        
    Hgen2 = {}
    for key in k2.keys():
        Hgen2[key] = m.Intermediate(Hrxn2[key] * rate2[key])
    
    Heat_Gen2 = m.Intermediate(-1* np.array(list(Hrxn2.values())).dot(np.array(list(rate2.values()))))
    
    #%% Heat Loss
    UA1 = m.FV(956393, lb=30000, ub=30000000)#3492914
    UA2 = m.FV(873221, lb=30000, ub=3e10)#4492914
    
    Heat_trans1 = m.Intermediate(UA1*(T_reactor1- T_ss1))
    Heat_trans2 = m.Intermediate(UA2*(T_reactor2- T_ss2))
    
    #%% NCp calculation
    # Cp function
    def Cp(a,b,c,d,T):
        Cp = a + (b/1000)*T + 100000*c/T**2 + (d/1000000)*T**2
        return Cp
    
    Cp_all = {}
    for key in keys:
        Cp_all[key] = Cp(data[key].A, data[key].B, data[key].C, data[key].D, T_st)
    
    # NCp (molar flowrate x Cp)
    NCp1 = m.Intermediate(np.array(list(Cp_all.values())).dot(np.array(list(yOut1.values()))))
    NCp2 = m.Intermediate(np.array(list(Cp_all.values())).dot(np.array(list(yOut2.values()))))
    
    ##%% Energy Balance
    #m.Equation(T_reactor1.dt() == (Heat_In1 - Heat_Out1 + Heat_Gen1 - Heat_trans1)/NCp1) # EB for Stage 1
    #m.Equation(T_reactor2.dt() == (Heat_In2 - Heat_Out2 + Heat_Gen2 - Heat_trans2)/NCp2) # EB for Stage 2
    
    m.Equation(T_reactor1.dt() == (Heat_In1 - Heat_Out1 + Heat_Gen1 - Heat_trans1)/91482)#84988) # EB for Stage 1
    m.Equation(T_reactor2.dt() == (Heat_In2 - Heat_Out2 + Heat_Gen2 - Heat_trans2)/90654)#84561) # EB for Stage 2
    
    
    #%% Solve 1 (steady-state mode for initial values for variables)
    
    # User Inputs
    T1_HI = 1100 # F
    T1_LO = 980 # F
    T2_HI = 1050 # F
    T2_LO = 970 # F 
    O2_HI = 20 # w%
    O2_LO = 10 # w%
    CO2_HI = 60 # w%
    CO2_LO = 0 # w%
    SO2_HI = 10 # w%
    SO2_LO = 0 # w%
    TCM_HI = 0.5 # w%
    TCM_LO = 0 # w%
    Sulf_HI = 0.05 # w%
    Sulf_LO = 0 # w%
    CO3_HI = 30 # w%
    CO3_LO = 15 # w%
    
    
    # MV validation data
    S2.value = data_tph["Sulf_in"][0]
    CO3.value = data_tph["CO3_in"][0]
    TCM.value = data_tph["Carbon_in"][0]
    Gold_in.value = data_tph["Gold_in"][0]
    Ore_in.value = data_tph["Ore_in"][0]
    Sul_in.value = data_tph["Sulfur_in"][0]
    O2_in.value = data_tph["O2_in"][0]
    
    # MV On/Off
    S2.FSTATUS = 0
    CO3.FSTATUS = 0
    TCM.FSTATUS = 0
    Gold_in.FSTATUS = 0
    Ore_in.FSTATUS = 0
    Sul_in.FSTATUS = 0
    O2_in.FSTATUS = 0
    
    S2.STATUS = 0
    CO3.STATUS = 0
    TCM.STATUS = 0
    Gold_in.STATUS = 0
    Ore_in.STATUS = 1
    Sul_in.STATUS = 1
    O2_in.STATUS = 1
    
    # CV On/Off
    T_reactor1.STATUS = 1
    T_reactor2.STATUS = 1
    wp_og1["O2"].STATUS = 1
    wp_og1["CO2"].STATUS = 1
    wp_og1["SO2"].STATUS = 1
    wp_calcine2["TCM"].STATUS = 1
    wp_calcine2["FeS2"].STATUS = 1
    wp_calcine2["CaCO3"].STATUS = 1
    
    # Read FV.NEWVAL from csv file for simulation (for IMODE = 1, 3 or 4)
    FVs = pd.read_csv(path + "FVs.csv", index_col=0)
    alpha_SO2.value = FVs["0"].alpha_SO2
    alpha_S.value = FVs["0"].alpha_S 
    alpha_Ore.value = FVs["0"].alpha_Ore
    alpha_Sulphate.value = FVs["0"].alpha_Sulphate
    alpha_TCM.value = FVs["0"].alpha_TCM
    air.value = FVs["0"].air
    O2_bias.value = FVs["0"].O2_bias
    TCM_bias.value = FVs["0"].TCM_bias
    CO3_bias.value = FVs["0"].CO3_bias
    Sulf_bias.value = FVs["0"].Sulf_bias
    UA1.value = FVs["0"].UA1
    UA2.value = FVs["0"].UA2
    
    #%% Solve 1 (Steady state mode for initial condition)
    # CV High Low 
    # T_reactor1.UPPER = (T1_HI - 32) * 5/9 + 273.15 # K
    # T_reactor1.LOWER = (T1_LO - 32) * 5/9 + 273.15 # K
    
    # T_reactor2.UPPER = (T2_HI - 32) * 5/9 + 273.15 # K
    # T_reactor2.LOWER = (T2_LO - 32) * 5/9 + 273.15 # K
    
    # wp_og1["O2"].UPPER = O2_HI # wt%
    # wp_og1["O2"].LOWER = O2_LO # wt%
    
    # wp_og1["CO2"].UPPER = CO2_HI # wt%
    # wp_og1["CO2"].LOWER = CO2_LO # wt%
    
    # wp_og1["SO2"].UPPER = SO2_HI # wt%
    # wp_og1["SO2"].LOWER = SO2_LO # wt%
    
    # wp_calcine2["TCM"].UPPER = TCM_HI # wt%
    # wp_calcine2["TCM"].LOWER = TCM_LO # wt%
    
    # wp_calcine2["FeS2"].UPPER = Sulf_HI # wt%
    # wp_calcine2["FeS2"].LOWER = Sulf_LO # wt%
    
    # wp_calcine2["CaCO3"].UPPER = CO3_HI # wt%
    # wp_calcine2["CaCO3"].LOWER = CO3_LO # wt%
    
    
    # MV High Low
    # Ore_in.UPPER = data_tph["Ore_in_HI"]
    # Ore_in.LOWER = data_tph["Ore_in_LO"]
    # Sul_in.UPPER = data_tph["Sul_in_HI"]
    # Sul_in.LOWER = data_tph["Sul_in_LO"]
    # O2_in.UPPER = data_tph["O2_in_HI"]
    # O2_in.LOWER = data_tph["O2_in_LO"]
    
    # MV Cost for Feed Maximization
    # Ore_in.COST = -1000 # '-' for maximization
    
    m.options.IMODE = 1
    m.options.SOLVER = 3
    m.solve(disp=True)
    
    
    Ore_in.value = data_tph["Ore_in"]
    Sul_in.value = data_tph["Sulfur_in"]
    
    
    m.options.IMODE = 4
    m.options.SOLVER = 3
    m.solve(disp=True)
    
    #%% Convert Kelvin to Fahrenheit
    T_1 = (np.array(T_reactor1.value) - 273.15) * 9/5 + 32 # F
    T_2 = (np.array(T_reactor2.value) - 273.15) * 9/5 + 32 # F
    
    #%% Convert tau (sec to min)
    tau1_m = np.array(tau1.value)/60
    tau2_m = np.array(tau2.value)/60
    
    #%% Gold outputs (Unit conversion from t/h to Oz/(ton of calcine) ) 
    Gold_calcine1_ozpt = np.array(wcalcine1["Au"]) * 35274 / np.array(wcalcine1_flowrate)
    Gold_calcine2_ozpt = np.array(wcalcine2["Au"]) * 35274 / np.array(wcalcine2_flowrate)
    
    #%% Unit conversions for outputs
    Ore_amps_new = (Ore_in.NEWVAL - 208.270) / 1.675786 # t/h to amps
    Sul_TPH_new = Sul_in.NEWVAL  # t/h (no need to convert)
    O2_scfm_new = O2_in.NEWVAL / 0.002421 # t/h to scfm
    
    #%% Calculation for MV High Low limits to download to OPC
    
    # MV ranges (+/- value from the RTO output)
    Ore_range = 3
    Sul_range = 1
    O2_range = 2
    
    # MV high low limits for download to OPC
    Ore_amps_HI = Ore_amps_new + Ore_range
    Ore_amps_LO = Ore_amps_new - Ore_range
    Sul_TPH_HI = Sul_TPH_new + Sul_range
    Sul_TPH_LO = Sul_TPH_new - Sul_range
    O2_scfm_HI = O2_scfm_new + O2_range
    O2_scfm_LO = O2_scfm_new - O2_range 
    
    
    #%%# Heat flow for Stage1 (to check the values for individual component) [J/s]
    H_In1_Ore = {}
    H_In1_Sulphur = {}
    H_In1_offgas = {}
    for key in keys_Ore:
        H_In1_Ore[key] = np.array(list(delH_In1_Ore[key]))*(np.array(list(yFeed_Ore[key])))
    for key in keys_Sulphur:
        H_In1_Sulphur[key] = np.array(list(delH_In1_Sulphur[key]))*(np.array(list(yFeed_Sulphur[key])))
    for key in keys_offgas:
        H_In1_offgas[key] = np.array(list(delH_In1_offgas[key]))*(np.array(list(yog2[key])))
        
    H_Out1_calcine = {}
    H_Out1_offgas = {}
    for key in keys_calcine:
        H_Out1_calcine[key] = np.array(list(delH_Out1_calcine[key]))*(np.array(list(ycalcine1[key])))
    for key in keys_offgas:
        H_Out1_offgas[key] = np.array(list(delH_Out1_offgas[key]))*(np.array(list(yog1[key])))
        
    H_In2_calcine = {}
    H_In2_feedgas = {}
    for key in keys_Ore:
        H_In2_calcine[key] = np.array(list(delH_In2_calcine[key]))*(np.array(list(ycalcine1[key])))
    for key in keys_feedgas:
        H_In2_feedgas[key] = np.array(list(delH_In2_feedgas[key]))*(np.array(list(yfeedgas[key])))
        
    H_Out2_calcine = {}
    H_Out2_offgas = {}
    for key in keys_Ore:
        H_Out2_calcine[key] = np.array(list(delH_Out2_calcine[key]))*(np.array(list(ycalcine2[key])))
    for key in keys_feedgas:
        H_Out2_offgas[key] = np.array(list(delH_Out2_offgas[key]))*(np.array(list(yog2[key])))
       
    # %% plotting
    t_minute = m.time/60
    
    plt.figure(1)
    plt.title('Feed rate')
    plt.subplot(3,1,1)
    plt.plot(t_minute,data_input["Ore_amps"], color='tab:blue', linestyle='-', linewidth=3,label='Ore_amps')
    plt.ylabel('Amps')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(3,1,2)
    plt.plot(t_minute,data_input["Sulfur_tph"], color='tab:blue', linestyle='-', linewidth=3,label='Sulfur_tph')
    plt.ylabel('t/h')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(3,1,3)
    plt.plot(t_minute,data_input["O2_scfm"], color='tab:blue', linestyle='-', linewidth=3,label='O2_scfm')
    plt.ylabel('SCFM[kg/h]')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    
    plt.figure(2)
    plt.title('Feed Ore Composition')
    plt.subplot(3,1,1)
    plt.plot(t_minute,data_input["Carbon_in"], color='tab:orange', linestyle='-', linewidth=3,label='Carbon_in (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(3,1,2)
    plt.plot(t_minute,data_input["Sulf_in"], color='tab:orange', linestyle='-', linewidth=3,label='Sulf_in (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')     
    plt.subplot(3,1,3)
    plt.plot(t_minute,data_input["CO3_in"], color='tab:orange', linestyle='-', linewidth=3,label='CO3_in (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')  
    
    # plt.figure(3)
    # plt.subplot(4,1,1)
    # plt.plot(t_minute,data_input["level1"], color='tab:orange', linestyle='-', width=3,label='level1')
    # plt.ylabel('%')
    # plt.xlabel('time[min]')
    # plt.legend(loc='best') 
    # plt.subplot(4,1,2)
    # plt.plot(t_minute,tau1_m, color='tab:green', linestyle='-', width=3,label='Residence time 1')
    # plt.ylabel('%')
    # plt.xlabel('time[min]')
    # plt.legend(loc='best') 
    # plt.subplot(4,1,3)
    # plt.plot(t_minute,data_input["level2"], color='tab:orange', linestyle='-', width=3,label='level2')
    # plt.ylabel('%')
    # plt.xlabel('time[min]')
    # plt.legend(loc='best') 
    # plt.subplot(4,1,4)
    # plt.plot(t_minute,tau2_m, color='tab:green', linestyle='-', width=3,label='Residence time 1')
    # plt.ylabel('%')
    # plt.xlabel('time[min]')
    # plt.legend(loc='best')  
    
    plt.figure(4)
    plt.title('Off Gas Composition')
    plt.subplot(3,1,1)
    plt.plot(t_minute,wp_og1["O2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_O2 (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(3,1,2)
    plt.plot(t_minute,wp_og1["CO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_CO2 (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')     
    plt.subplot(3,1,3)
    plt.plot(t_minute,wp_og1["SO2"], color='tab:blue', linestyle='-', linewidth=3,label='og1_SO2 (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')     
    
    plt.figure(5)
    plt.title('Calcine Composition')
    plt.subplot(3,1,1)
    plt.plot(t_minute,wp_calcine2["TCM"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_TCM (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(3,1,2)
    plt.plot(t_minute,wp_calcine2["FeS2"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_Sulfur (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')     
    plt.subplot(3,1,3)
    plt.plot(t_minute,wp_calcine2["CaCO3"], color='tab:red', linestyle='-', linewidth=3,label='Calcine2_CO3 (wt%)')
    plt.ylabel('wt%')
    plt.xlabel('time[min]')
    plt.legend(loc='best')  
    
    plt.figure(6)
    plt.title("Reactor Temperature")
    plt.subplot(2,1,1)
    plt.plot(t_minute,T_1, color='tab:red', linestyle='-', linewidth=3,label='T1')
    plt.ylabel('F')
    plt.xlabel('time[min]')
    plt.legend(loc='best') 
    plt.subplot(2,1,2)
    plt.plot(t_minute,T_2, color='tab:red', linestyle='-', linewidth=3,label='T2')
    plt.ylabel('F')
    plt.xlabel('time[min]')
    plt.legend(loc='best')     
    
    plt.show()
    
    #%% save data to csv
    df = pd.DataFrame()
    df['Time'] = t_minute
    df['Ore_amps'] = data_input["Ore_amps"]
    df['Sulfur_tph'] = data_input["Sulfur_tph"]
    df['O2_scfm'] = data_input["O2_scfm"]
    df['Carbon_in'] = data_input["Carbon_in"]
    df['Sulf_in'] = data_input["Sulf_in"]
    df['CO3_in'] = data_input["CO3_in"]
    
    df['O2'] = wp_og1["O2"]
    df['CO2'] = wp_og1["CO2"]
    df['SO2'] = wp_og1["SO2"]
    df['TCM'] = wp_calcine2["TCM"]
    df['FeS2'] = wp_calcine2["FeS2"]
    df['CaCO3'] = wp_calcine2["CaCO3"]
    
    df['T1'] = T_1
    df['T2'] = T_2
    df['Ore_amps'] = data_input["Ore_amps"]
    
    df.to_csv('Roaster_data_training_random'+str(num[i])+'.csv')