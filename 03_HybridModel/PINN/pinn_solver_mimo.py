import tensorflow as tf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set data type
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


class PINNSolver:
    def __init__(self, model=None, is_pinn=True, *args, **kwargs):
        self.model = model

        # Initialize history of losses and global iteration counter
        self.current_loss = 0
        self.hist = []
        self.iter = 0
        self.hello_pinn_info = 'Hello PINN!'
        self.is_pinn = is_pinn
        self.loss1 = 0
        self.loss2 = 0
        self.loss3 = 0

    def get_residual_loss(self):
        return self.residual_function()

    def objective_function(self, x, y):

        # Add neural network loss
        y_pred = self.model(x)

        # Initialize loss
        self.loss1 = tf.reduce_mean(tf.square(y - y_pred))

        if self.is_pinn:
            # Compute residuals
            residual_loss1, residual_loss2 = self.get_residual_loss()
#             res_loss_mse = tf.reduce_mean(tf.square(residual_loss))
            self.loss2 = tf.reduce_mean(tf.square(residual_loss1))
            self.loss3 = tf.reduce_mean(tf.square(residual_loss2))
            # tf.print(self.loss2)
            # loss = self.loss2

            # loss = self.loss1 + self.loss2
            loss = self.loss1 + self.loss2 + self.loss3

        else:
            loss = self.loss1

        return loss

    def get_grad(self, x, y):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss = self.objective_function(x, y)

        loss_grad = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, loss_grad

    def residual_function(self, *args):
        """Residual of the PDE"""
        self.hello_pinn_info = "Goodbye PINN!"
        return 0

    def solve_with_tf_optimizer(self, optimizer=tf.keras.optimizers.Adam(), x=None, y=None, n_step=1001):
        """This method performs a gradient descent type optimization."""
        best = 1e20
        wait = 0
        patience = 50

        @tf.function
        def train_step():
            loss, loss_grad = self.get_grad(x, y)

            # Perform gradient descent step
            optimizer.apply_gradients(zip(loss_grad, self.model.trainable_variables))
            return loss

        for i in range(n_step):
            loss = train_step().numpy()

            self.current_loss = loss
            self.callback()

            wait += 1
            if loss < best:
                best = loss
                wait = 0
            if wait >= patience:
                break

    def solve_with_scipy_optimizer(self, x, u, method='L-BFGS-B', **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""

        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            weight_list = []
            shape_list = []

            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())

            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape

                # Weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw
                    
                elif len(vs) == 3:
                    sw = vs[0] * vs[1] * vs[2]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1], vs[2]))
                    idx += sw


                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx + vs[0]]
                    idx += vs[0]

                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1

                ''' Error should raise if vs > 2: not implemented yet! '''

                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))

        def get_loss_and_grad(weights):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""

            # Update weights in model
            set_weight_tensor(weights)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, loss_grad = self.get_grad(x, u)

            # Store current loss for callback function
            loss = loss.numpy().astype(np.float64)
            self.current_loss = loss

            # Flatten gradient
            grad_flat = []
            for g in loss_grad:
                grad_flat.extend(g.numpy().flatten())

            # Gradient list to array
            grad_flat = np.array(grad_flat, dtype=np.float64)

            # Return value and gradient of \phi as tuple
            return loss, grad_flat

        return minimize(fun=get_loss_and_grad,
                        x0=x0,
                        jac=True,
                        method=method,
                        callback=self.callback,
                        options={'maxiter': 5,
                                 'maxfun': 5000000,
                                 'maxcor': 5,
                                 'maxls': 5,
                                 'ftol': 1.0 * np.finfo(float).eps},
                        **kwargs)

    def callback(self, xr=None):
        if self.iter % 10 == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss))
        self.hist.append(self.current_loss)
        self.iter += 1
