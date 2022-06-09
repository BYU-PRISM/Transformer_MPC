import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import time


# Road data to check
data = pd.read_pickle('TCLab_PINN_On_MIMO_Control_Multi_Trans_20_20_1_1_1hr.pkl')
# data = data[0:3600]

plt.figure(0)
plt.subplot(2, 1, 1)
plt.plot(data["T1"], 'c-', label='T1')
plt.plot(data["T2"], 'm-', label='T2')
plt.step(data["SP1"], 'r--', label='SP1')
plt.step(data["SP2"], 'g--', label='SP2')
plt.legend(loc=2)

plt.subplot(2, 1, 2)
plt.step(data["H1"], 'r-', label='H1')
plt.step(data["H2"], 'g-', label='H2')
plt.legend(loc=2)

plt.tight_layout()

plt.savefig('TCLab_Control_onestep_lstm_1hr.eps', format='eps')
plt.savefig('TCLab_Control_onestep_lstm_1hr.png')

plt.show()

