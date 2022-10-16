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
    ref = np.ones(y.shape)
    return  cp.norm(y-ref,'fro')**2

# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of input/output constraints
    # no real constraints on y, input should be between -1 and 1
    return [u >= -1, u <= 1]

# DeePC paramters
s = 1                       # How many steps before we solve again the DeePC problem
T_INI = 2                   # Size of the initial set of data
T_list = [100]              # Number of data points used to estimate the system
HORIZON = 10                # Horizon length
LAMBDA_G_REGULARIZER = 0    # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0    # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0    # u regularizer
EXPERIMENT_HORIZON = 100    # Total number of steps

# model of two-tank example
A = np.array([
        [0.70469, 0.     ],
        [0.24664, 0.70469]])
B = np.array([[0.75937], [0.12515]])
C = np.array([[0., 1.]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=1))

fig, ax = plt.subplots(1,2)
plt.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f'Simulating with {T} initial samples...')
    sys.reset()
    # Generate initial data and initialize DeePC
    
    data = sys.apply_input(u = np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
    sys.reset(data_ini = data_ini)

    deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = LAMBDA_G_REGULARIZER,
        lambda_y = LAMBDA_Y_REGULARIZER,
        lambda_u = LAMBDA_U_REGULARIZER)

    for _ in range(EXPERIMENT_HORIZON//s):
        # Solve DeePC
        u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True)

        # Apply optimal control input
        _ = sys.apply_input(u = u_optimal[:s, :], noise_std=1e-2)

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')
    ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')

ax[0].set_ylim(0, 1.5)
ax[1].set_ylim(-1.2, 1.2)
ax[0].set_xlabel('t')
ax[0].set_ylabel('y')
ax[0].grid()
ax[1].set_ylabel('u')
ax[1].set_xlabel('t')
ax[1].grid()
ax[0].set_title('Closed loop - output signal $y_t$')
ax[1].set_title('Closed loop - control signal $u_t$')
plt.legend(fancybox=True, shadow=True)
plt.show()