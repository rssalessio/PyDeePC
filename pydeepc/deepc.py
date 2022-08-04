import numpy as np
from typing import Tuple, Callable, List, Optional, Union, Dict
import cvxpy as cp
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc.utils import Data, split_data, create_hankel_matrix

class DeePC(object):
    def __init__(self, data: Data, Tini: int, horizon: int):
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        :param Tini:                number of samples needed to estimate initial conditions
        :param horizon:             horizon length
        """
        self.Tini = Tini
        self.horizon = horizon
        self.update_data(data)

    def update_data(self, data: Data):
        """
        Update Hankel matrices of DeePC
        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], "Input/output data must have the same length"
        assert data.y.shape[0] - self.Tini - self.horizon + 1 >= 1, f"There is not enough data: this value {data.y.shape[0] - self.Tini - self.horizon + 1} needs to be >= 1"

        Up, Uf, Yp, Yf = split_data(data, self.Tini, self.horizon)

        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        
        self.M = data.u.shape[1]
        self.P = data.y.shape[1]
        self.T = data.u.shape[0]


    def solve_deepc(
            self,
            data_ini: Data,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            g_regularizer: float = 0.,
            y_regularizer: float = 0.) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data_ini:            A tuple of input/output data used to estimate initial condition.
                                    Data should have shape Tini x M where Tini is the batch size and
                                    M is the number of features
        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :param g_regularizer:       non-negative scalar. Regularization factor for g. Used for
                                    stochastic/non-linear systems.
        :param y_regularizer:       non-negative scalar. Regularization factor for y. Used for
                                    stochastic/non-linear systems.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['u']: u decision variable
                                    info['y']: y decision variable
                                    info['value']: value of the optimization problem
                                    info['g']: value of g
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(data_ini.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data_ini.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data_ini.u.shape[1] == self.M, "Incorrect number of features for the input signal"
        assert data_ini.y.shape[1] == self.P, "Incorrect number of features for the output signal"
        assert data_ini.y.shape[0] == data_ini.u.shape[0], "Input/output data must have the same length"
        assert data_ini.y.shape[0] == self.Tini, f"Invalid size"
        assert build_loss is not None, "Loss function callback cannot be none"
        assert g_regularizer >= 0 and y_regularizer >= 0, "Regularizers must be non-negative"


        # Need to transpose to make sure that time is over the columns, and features over the rows
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        # Build variables
        u = cp.Variable(shape=(self.M * self.horizon))
        y = cp.Variable(shape=(self.P * self.horizon))
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1))
        sigma_y = cp.Variable(shape=(self.Tini * self.P))

        A = cp.vstack([self.Up, self.Yp, self.Uf, self.Yf])
        b = cp.hstack([uini, yini + sigma_y, u, y])

        # Build constraints
        constraints = [A @ g == b]

        u = cp.reshape(u, (self.horizon, self.M))
        y = cp.reshape(y, (self.horizon, self.P))
        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)
        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        _regularizers = g_regularizer * cp.norm(g, p=1) + y_regularizer * cp.norm(sigma_y, p=1)
        assert _loss is not None, "Invalid loss"

        objective = cp.Minimize(_loss + _regularizers)
        problem = cp.Problem(objective, constraints)

        # Solve problem
        result = problem.solve()
        u_optimal = (self.Uf @ g.value).reshape(self.horizon, self.M)
        info = {'value': result, 'u': u.value, 'y': y.value, 'g': g.value, 'u_optimal': u_optimal}

        return u_optimal, info
