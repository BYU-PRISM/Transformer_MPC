import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_pickle('Roaster_data_training_random_test.pkl')
data2 = pd.read_pickle('Roaster_data_training_random10.pkl')


plt.figure()
plt.plot(data1["TCM"])
plt.plot(data2["TCM"], '--')

plt.figure()
plt.plot(data1["Ore_amps"])
plt.plot(data2["Ore_amps"])
plt.show()