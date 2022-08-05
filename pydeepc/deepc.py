from cmath import isinf
import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc.utils import (
    Data,
    split_data,
    low_rank_matrix_approximation,
    OptimizationProblem,
    OptimizationProblemVariables)

class DeePC(object):
    optimization_problem: OptimizationProblem = None

    def __init__(self, data: Data, Tini: int, horizon: int):
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        :param Tini:                number of samples needed to estimate initial conditions
        :param horizon:             horizon length
        :param explained_variance:  Regularization term in (0,1] used to approximate the Hankel matrices.
                                    By default is None (no low-rank approximation is performed).
        """
        self.Tini = Tini
        self.horizon = horizon
        self.update_data(data)

        self.optimization_problem = None

    def update_data(self, data: Data):
        """
        Update Hankel matrices of DeePC. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], \
            "Input/output data must have the same length"
        assert data.y.shape[0] - self.Tini - self.horizon + 1 >= 1, \
            f"There is not enough data: this value {data.y.shape[0] - self.Tini - self.horizon + 1} needs to be >= 1"
        
        Up, Uf, Yp, Yf = split_data(data, self.Tini, self.horizon)

        self.Up = Up
        self.Uf = Uf
        self.Yp = Yp
        self.Yf = Yf
        
        self.M = data.u.shape[1]
        self.P = data.y.shape[1]
        self.T = data.u.shape[0]

        self.optimization_problem = None

    def build_problem(self,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            lambda_g: float = 0.,
            lambda_y: float = 0.,
            explained_variance: Optional[float] = None) -> OptimizationProblem:
        """
        Builds the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf


        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :param lambda_g:            non-negative scalar. Regularization factor for g. Used for
                                    stochastic/non-linear systems.
        :param lambda_y:            non-negative scalar. Regularization factor for y. Used for
                                    stochastic/non-linear systems.
        :param explained_variance:  Regularization term in (0,1] used to approximate the Hankel matrices.
                                    By default is None (no low-rank approximation is performed).
        :return:                    Parameters of the optimization problem
        """
        assert build_loss is not None, "Loss function callback cannot be none"
        assert lambda_g >= 0 and lambda_y >= 0, "Regularizers must be non-negative"
        assert explained_variance is None or explained_variance > 0 and explained_variance <= 1., \
            "Explained variance should be None or a value in (0,1]"

        self.optimization_problem = False

        # Build variables
        uini = cp.Parameter(shape=(self.M * self.Tini))
        yini = cp.Parameter(shape=(self.M * self.Tini))
        u = cp.Variable(shape=(self.M * self.horizon))
        y = cp.Variable(shape=(self.P * self.horizon))
        g = cp.Variable(shape=(self.T - self.Tini - self.horizon + 1))
        sigma_y = cp.Variable(shape=(self.Tini * self.P))

        Up, Yp, Uf, Yf = self.Up, self.Yp, self.Uf, self.Yf

        if explained_variance is not None:
            G = low_rank_matrix_approximation(np.vstack([Up, Yp, Uf, Yf]), explained_var=explained_variance)
            idxs = np.cumsum([Up.shape[0], Yp.shape[0], Uf.shape[0], Yf.shape[0]]).astype(np.int64)
            Up, Yp = G[:idxs[0]], G[idxs[0] : idxs[1]]
            Uf, Yf = G[idxs[1] : idxs[2]], G[idxs[2] : idxs[3]]

        A = cp.vstack([Up, Yp, Uf, Yf])
        b = cp.hstack([uini, yini + sigma_y, u, y])

        # Build constraints
        constraints = [A @ g == b]

        # u, y = self.Uf @ g, self.Yf @ g
        u = cp.reshape(u, (self.horizon, self.M))
        y = cp.reshape(y, (self.horizon, self.P))

        _constraints = build_constraints(u, y) if build_constraints is not None else (None, None)

        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)

        # Build loss
        _loss = build_loss(u, y)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        _regularizers = lambda_g * cp.norm(g, p=1) + lambda_y * cp.norm(sigma_y, p=1)
        problem_loss = _loss + _regularizers

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(u_ini = uini, y_ini = yini, u = u, y = y, g = g, sigma_y = sigma_y),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem

    def solve(
            self,
            data_ini: Data,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param data_ini:            A tuple of input/output data used to estimate initial condition.
                                    Data should have shape Tini x M where Tini is the batch size and
                                    M is the number of features
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert len(data_ini.u.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data_ini.y.shape) == 2, "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data_ini.u.shape[1] == self.M, "Incorrect number of features for the input signal"
        assert data_ini.y.shape[1] == self.P, "Incorrect number of features for the output signal"
        assert data_ini.y.shape[0] == data_ini.u.shape[0], "Input/output data must have the same length"
        assert data_ini.y.shape[0] == self.Tini, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        # Need to transpose to make sure that time is over the columns, and features over the rows
        uini, yini = data_ini.u[:self.Tini].flatten(), data_ini.y[:self.Tini].flatten()

        self.optimization_problem.variables.u_ini.value = uini
        self.optimization_problem.variables.y_ini.value = yini

        try:
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        u_optimal = (self.Uf @ self.optimization_problem.variables.g.value).reshape(self.horizon, self.M)
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info
