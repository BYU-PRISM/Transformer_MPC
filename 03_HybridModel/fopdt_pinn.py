from PINN.pinn_solver import PINNSolver
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import time
import pandas as pd


# For LSTM model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model
from tqdm.keras import TqdmCallback


DTYPE = 'float32'
tf.random.set_seed(12345)
np.random.seed(12345)


class SPINN(PINNSolver):
    def __init__(self, x_r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store Model Variables
        self.K = 1.0
        self.tau = 2.0
        # Store Model Constants

        # Store collocation points
        self.Xs = x_r[:]
        
    @tf.function
    def get_residual_loss(self):
    #     with tf.GradientTape(persistent=True) as tape:
    #         # Watch variables representing t and x during this GradientTape
    #         # tape.watch(self.t)

    #         # Compute current values y(t,x)
        y = self.model(self.Xs[:])
            
    #         # y_t = tape.gradient(y, self.t)
    #     # y_tt = tape.gradient(y_t, self.t)

    #     del tape

        return self.residual_function(self.u, y)

    def residual_function(self, u, y):
        """Residual of the PDE"""
        # res = y_tt + self.mu*y_t + self.k*y
        # y = self.model(self.Xs[:])
        y_t = (y[1:]-y[:-1]) / self.dt
        res = self.tau * y_t + y - self.K * self.u
        # res[0] -= qh
        # return y_t/3000.0*100.0 - self.model.alpha * y_xx/(0.1-0.001)**2*100 + self.model.beta * (y*100+20 - self.Ts)
        return 1e-2*res

    def callback(self, *args):
        if self.iter % 10 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss)
                  )
        self.hist.append(self.current_loss)
        self.iter += 1


# Load Data
data = pd.read_pickle('open_loop_data_SISO.pkl')

window = 5

# Scale features
s1 = MinMaxScaler(feature_range=(-1,1))
Xs = s1.fit_transform(data[['u','y']].to_numpy())
s2 = MinMaxScaler(feature_range=(-1,1))
Ys = s2.fit_transform(data[['y']].to_numpy())

# Save model parameters
model_params = dict()
model_params['Xscale'] = s1
model_params['yscale'] = s2
model_params['window'] = window

dump(model_params, open(path + 'model_param.pkl', 'wb'))



# def oscillator(d, w0, x):
#     """Defines the analytical solution to the 1D underdamped harmonic oscillator problem.
#     Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
#     assert d < w0
#     w = np.sqrt(w0 ** 2 - d ** 2)
#     phi = np.arctan(-d / w)
#     A = 1 / (2 * np.cos(phi))
#     cos = tf.cos(phi + w * x)
#     sin = tf.sin(phi + w * x)
#     exp = tf.exp(-d * x)
#     y = exp * 2 * A * cos
#     return y


# d, w0 = 2, 20
# x = tf.linspace(0, 1, 500)
# y = oscillator(d, w0, x)
# print(x.shape, y.shape)

# slice out a small number of points from the LHS of the domain
x_data = x[0:200:20][:, None]
y_data = y[0:200:20][:, None]
print(x_data.shape, y_data.shape)


# # Initialize LSTM model
model_lstm = Sequential()

model_lstm.add(LSTM(units=100, return_sequences=True, \
          input_shape=(X_train.shape[1],X_train.shape[2])))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=100, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(units=100))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(units=Y_train.shape[1])) #units = number of outputs
model_lstm.compile(optimizer = 'adam', loss = 'mean_squared_error',\
              metrics = ['accuracy'])
# Allow for early exit
es_lstm = EarlyStopping(monitor='loss',mode='min',verbose=1,patience=10)

# # Fit (and time) LSTM model
# t0 = time.time()
# result_lstm = model_lstm.fit(X_train, Y_train, epochs = 300, batch_size = 250,\
#                              callbacks=[es_lstm, TqdmCallback(verbose=1)],\
#                              verbose=0, validation_data=(X_val, Y_val))
# t1 = time.time()
# print('Runtime: %.2f s' %(t1-t0))


# layers = [1, 32, 32, 32, 1]

# # Input Layer
# model.add(tf.keras.layers.InputLayer(input_shape=(layers[0])))

# # Hidden Layers
# for n_i in range(1, len(layers) - 1):
#     model.add(tf.keras.layers.Dense(units=layers[n_i],
#                                     activation='tanh',
#                                     kernel_initializer='glorot_normal')
#               )

# model.add(tf.keras.layers.Dense(units=layers[-1], kernel_initializer='glorot_normal')
#           )

x_physics = x[::5][:, None]
y_physics = y[::5][:, None]
x_physics = tf.convert_to_tensor(x_physics.numpy(), dtype=DTYPE)
y_physics = tf.convert_to_tensor(y_physics.numpy(), dtype=DTYPE)

solver = SPINN(model=model, x_r=x_physics, is_pinn=True)

# # Choose step sizes aka learning rate
# lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 5000], [1e-3, 1e-4, 1e-5])

# Solve with Adam optimizer
optim = tf.keras.optimizers.Adam()

x_data = tf.convert_to_tensor(x_data.numpy(), DTYPE)
y_data = tf.convert_to_tensor(y_data.numpy(), DTYPE)

solver.solve_with_tf_optimizer(optim, x_data, y_data, n_step=5000)
solver.is_pinn = True
solver.solve_with_tf_optimizer(optim, x_physics, y_physics, n_step=18000)

solver.solve_with_scipy_optimizer(x_physics, y_physics, method='L-BFGS-B')
# solver.solve_with_scipy_optimizer(x_physics, y_physics, method='SLSQP')

yp = model(x)

plt.plot(x, yp)
plt.plot(x, y, '-.')
plt.show()
