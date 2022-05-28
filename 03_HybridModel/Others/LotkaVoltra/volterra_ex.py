import pickle
from pickle import dump, load
from PINN.pinn_solver import PINNSolver
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

DTYPE = 'float32'
tf.random.set_seed(12345)
np.random.seed(12345)


class LVPINN(PINNSolver):
    def __init__(self, x_r, param, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Store Model Variables
        # self.model.alpha = tf.Variable(initial_value=8e-6, trainable=True, dtype=DTYPE, constraint=tf.keras.constraints.NonNeg())
        # self.model.beta = tf.Variable(initial_value=2e-4, trainable=True, dtype=DTYPE, constraint=tf.keras.constraints.NonNeg())

        # Store Model Constants
        # self.alpha = None
        # self.beta = None
        # self.Ts = None

        # Store collocation points
        # self.x = x_r[:, 0:1]
        self.t = x_r[:, 0:1]

        self.R = param[0]
        self.U = param[1]

        self.temp1 = None
        self.temp2 = None
        # self.qh = x_r[:, 2:3]

        # self.x_u = self.x*(x_ub-x_lb) + x_lb
        # self.t_u = self.t*(t_ub-t_lb)+t_lb

    @tf.function
    def get_residual_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and x during this GradientTape
            tape.watch(self.t)
            # tape.watch(self.x)
            # tape.watch(self.qh)

            # Compute current values y(t,x)
            y1 = self.model(self.t[:, 0])[0]
            y2 = self.model(self.t[:, 0])[1]

        y_t1 = tape.gradient(y1, self.t)
        y_t2 = tape.gradient(y2, self.t)
        self.temp1 = y1
        self.temp2 = y_t2
        del tape

        return self.residual_function(self.t, y1, y2, y_t1, y_t2)

    def residual_function(self, t, y1, y2, y_t1, y_t2):
        """Residual of the PDE"""
        # print(y_t)
        r = y1
        dr_dt = y_t1
        p = y2
        dp_dt = y_t2

        R = self.R
        U = self.U

        res1 = dr_dt - R / U * (2 * U * r - 0.04 * U ** 2 * r * p)
        res2 = dp_dt - R / U * (0.02 * U ** 2 * r * p - 1.06 * U * p)
        res = res1 + res2
        # res = tf.reduce_sum(res)
        # print(r)
        return 1e-4*res

    def callback(self, *args):
        if self.iter % 10 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss)
                  )
        self.hist.append(self.current_loss)
        self.iter += 1


with open('volterra.pkl', 'rb') as file:
    [t, y, R, U] = load(file)
    file.close()

n_t = len(t)

N_u = 400

layers = [1, 10, 10, 10, 10, 5, 2]

t = t.flatten()[:, None]

Exact = np.real(y).T

X_star = t
# u_star = Exact.flatten()[:, None]
u_star = Exact.T

idx = np.random.choice(X_star.shape[0], N_u, replace=False)
# X_u_train = X_star[idx, :]
# u_train = u_star[idx, :]
X_u_train = X_star
u_train = u_star

X_u_train = tf.convert_to_tensor(X_u_train, dtype=DTYPE)
u_train = tf.convert_to_tensor(u_train, dtype=DTYPE)

model = tf.keras.Sequential()
# Input Layer
model.add(tf.keras.layers.InputLayer(input_shape=(layers[0])))

# Hidden Layers
for n_i in range(1, len(layers) - 3):
    model.add(tf.keras.layers.Dense(units=layers[n_i],
                                    activation='tanh',
                                    kernel_initializer='glorot_normal')
              # kernel_initializer='glorot_uniform')
              )
for n_i in range(len(layers) - 3, len(layers) - 1):
    model.add(tf.keras.layers.Dense(units=layers[n_i],
                                    activation='sigmoid',
                                    kernel_initializer='glorot_normal')
              # kernel_initializer='glorot_uniform')
              )

model.add(tf.keras.layers.Dense(units=layers[-1], kernel_initializer='glorot_normal')
          # kernel_initializer='glorot_uniform')
          )


solver = LVPINN(model=model, x_r=X_u_train, is_pinn=False, param=(R, U))
# Choose step sizes aka learning rate
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 5000], [1e-3, 1e-4, 1e-5])

# Solve with Adam optimizer
optim = tf.keras.optimizers.Adam()

# Start timer
# t0 = time()
solver.solve_with_tf_optimizer(optim, X_u_train, u_train, n_step=50000)
solver.solve_with_scipy_optimizer(X_u_train, u_train, method='SLSQP')


yp = model(X_star)
yp = yp.numpy()

plt.figure()
# plt.plot(t, Th.value, 'r-', label=r'$T_{heater}\,(^oC)$')
plt.plot(t, Exact[0, :], '-', label='Prey')
plt.plot(t, Exact[1, :], '-', label='Predator')

plt.plot(t, yp[:, 0], '--', label='$ \hat{Prey} $')
plt.plot(t, yp[:, 1], '--', label='$ \hat{Predator} $')

plt.ylabel('Relative Population')
plt.xlabel('Time')
# plt.xlim([0, 50])
plt.legend(loc=4)
plt.show()


