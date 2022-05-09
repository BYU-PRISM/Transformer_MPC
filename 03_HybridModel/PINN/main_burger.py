from pinn_solver import PINNSolver
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
from plot_fun import my_plot


DTYPE = 'float32'
tf.random.set_seed(1234)
np.random.seed(1234)


class MyPINN(PINNSolver):
    def __init__(self, x_r, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store Model Variables
        self.model.l1 = tf.Variable(initial_value=0.0, trainable=True, dtype=DTYPE)
        self.model.l2 = tf.Variable(initial_value=-6.0, trainable=True, dtype=DTYPE)

        # Store collocation points
        self.x = x_r[:, 0:1]
        self.t = x_r[:, 1:2]

    def get_residual_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and x during this GradientTape
            tape.watch(self.t)
            tape.watch(self.x)

            # Compute current values y(t,x)
            y = self.model(tf.stack([self.x[:, 0], self.t[:, 0]], axis=1))
            y_x = tape.gradient(y, self.x)

        y_t = tape.gradient(y, self.t)
        y_xx = tape.gradient(y_x, self.x)

        del tape

        return self.residual_function(self.t, self.x, y, y_t, y_x, y_xx)

    def residual_function(self, t, x, y, y_t, y_x, y_xx):
        """Residual of the PDE"""
        return y_t + self.model.l1 * y * y_x - tf.exp(self.model.l2) * y_xx

    def callback(self, *args):
        if self.iter % 10 == 0:
            print('It {:05d}: loss = {:10.8e}, l1 = {:10.4}, l2 = {:10.4}'.format(self.iter,
                                                                                  self.current_loss,
                                                                                  self.model.l1.value().numpy(),
                                                                                  np.exp(self.model.l2.value().numpy()))
                  )
        self.hist.append(self.current_loss)
        self.iter += 1


nu = 0.01 / np.pi

N_u = 2000

layers = [2, 5, 10, 10, 7, 4, 1]

data = loadmat('../Data/burgers_shock.mat')

t = data['t'].flatten()[:, None]
x = data['x'].flatten()[:, None]
Exact = np.real(data['usol']).T
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
u_star = Exact.flatten()[:, None]
# Doman bounds
lb = X_star.min(0)
ub = X_star.max(0)
######################################################################
######################## Noiseles Data ###############################
######################################################################
noise = 0.0

idx = np.random.choice(X_star.shape[0], N_u, replace=False)
X_u_train = X_star[idx, :]
u_train = u_star[idx, :]

X_u_train = tf.convert_to_tensor(X_u_train, dtype=DTYPE)
u_train = tf.convert_to_tensor(u_train, dtype=DTYPE)


model = tf.keras.Sequential()
# Input Layer
model.add(tf.keras.layers.InputLayer(input_shape=(layers[0])))

# Hidden Layers
for n_i in range(1, len(layers) - 1):
    model.add(tf.keras.layers.Dense(units=layers[n_i],
                                    activation='tanh',
                                    kernel_initializer='glorot_normal')
              # kernel_initializer='glorot_uniform')
              )

model.add(tf.keras.layers.Dense(units=layers[-1], kernel_initializer='glorot_normal')
          # kernel_initializer='glorot_uniform')
          )

# model.build((None, 2))
solver = MyPINN(model=model, x_r=X_u_train)

# Choose step sizes aka learning rate
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 5000], [1e-1, 1e-2, 1e-3])

# Solve with Adam optimizer
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Start timer
t0 = time()
solver.solve_with_tf_optimizer(optim, X_u_train, u_train, n_step=15000)
solver.solve_with_scipy_optimizer(X_u_train, u_train, method='L-BFGS-B')
# solver.solve_with_scipy_optimizer(X_u_train, u_train, method='SLSQP')

# Print computation time
print('\nComputation time: {} seconds'.format(time() - t0))

u_pred = model(X_star)
my_plot(u_pred.numpy(), X_u_train.numpy(), u_train.numpy(), t, x, Exact, X_star, X, T)