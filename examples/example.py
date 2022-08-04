# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc import DeePC
from pydeepc.utils import Data
from utils import System

# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = 1
    # Sum_t ||y_t - r_t||^2
    return cp.sum(cp.norm(y - ref, p=2, axis=1))  # cp.sum(cp.norm(u, p=2, axis=1))

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    return [y <= 10, y >= -10, u >= -20, u <= 20]

# DeePC paramters
s = 3                       # How many steps before we solve again the DeePC problem
T_INI = 5                   # Size of the initial set of data
T_list = [100, 150, 200]    # Number of data points used to estimate the system
HORIZON = 30                # Horizon length
LAMBDA_G_REGULARIZER = 0    # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0    # y regularizer (see DeePC paper, eq. 8)

# Plant
# In this example we consider the three-pulley 
# system analyzed in the original VRFT paper:
# 
# "Virtual reference feedback tuning: 
#      a direct method for the design offeedback controllers"
# -- Campi et al. 2003

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = System(scipysig.TransferFunction(num, den, dt=dt).to_ss())

plt.figure()
plt.margins(x=0, y=0)

# Simulate for different values of T
for T in T_list:
    sys.reset()
    # Generate initial data and initialize DeePC
    data = sys.apply_input(u = np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
    sys.reset(data_ini = data_ini)

    for idx in range(300):
        # Solve DeePC
        u_optimal, info = deepc.solve_deepc(
            data_ini = data_ini,
            build_loss = loss_callback,
            build_constraints = constraints_callback,
            g_regularizer = LAMBDA_G_REGULARIZER,
            y_regularizer = LAMBDA_Y_REGULARIZER)

        # Apply optimal control input
        _ = sys.apply_input(u = u_optimal[:s, :], noise_std=0)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    plt.plot(data.y[T:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')


plt.xlabel('Step')
plt.ylabel('y')
plt.title('Closed loop output')
plt.legend(fancybox=True, shadow=True)
plt.grid()
plt.show()