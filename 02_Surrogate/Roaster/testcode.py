import matplotlib.pyplot as plt
import pandas as pd

data1 = pd.read_pickle('Roaster_data_training_random_test.pkl')
data2 = pd.read_pickle('Roaster_data_training_random10.pkl')


plt.figure()
plt.plot(data1["O2"])
plt.plot(data2["O2"], '--')

plt.figure()
plt.plot(data1["T1"])
plt.plot(data2["T1"])


plt.show()