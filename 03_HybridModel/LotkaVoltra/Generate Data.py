import pickle
from pickle import dump, load
# from PINN.pinn_solver import PINNSolver
# import tensorflow as tf
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt


def lotka_volterra(x, t, U, R):
    dx = [0, 0]

    r = x[0]
    p = x[1]

    dx[0] = (R / U) * (2 * U * r - 0.04 * U ** 2 * r * p)
    dx[1] = (R / U) * (0.02 * U ** 2 * r * p - 1.06 * U * p)

    return dx


t = np.linspace(0, 1, 501)

U = 200
R = 20

y0 = [100 / U, 15 / U]

y = integrate.odeint(lotka_volterra, y0, t, args=(U, R))

plt.plot(t, y, label=['Prey', 'Predator'])

with open('volterra.pkl', 'wb') as file:
    pickle.dump([t, y, R, U], file)

    file.close()
