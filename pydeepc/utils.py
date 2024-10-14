import numpy as np
from typing import NamedTuple, Tuple, Optional, List, Union
from cvxpy import Expression, Variable, Problem, Parameter
from cvxpy.constraints.constraint import Constraint
from numpy.typing import NDArray

class OptimizationProblemVariables(NamedTuple):
    """
    Class used to store all the variables used in the optimization
    problem
    """
    u_ini: Union[Variable, Parameter]
    y_ini: Union[Variable, Parameter]
    u: Union[Variable, Parameter]
    y: Union[Variable, Parameter]
    g: Union[Variable, Parameter]
    slack_y: Union[Variable, Parameter]
    slack_u: Union[Variable, Parameter]


class OptimizationProblem(NamedTuple):
    """
    Class used to store the elements an optimization problem
    :param problem_variables:   variables of the opt. problem
    :param constraints:         constraints of the problem
    :param objective_function:  objective function
    :param problem:             optimization problem object
    """
    variables: OptimizationProblemVariables
    constraints: List[Constraint]
    objective_function: Expression
    problem: Problem

class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param y: output data
    """
    u: NDArray[np.float64]
    y: NDArray[np.float64]


def create_hankel_matrix(data: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """
    Create an Hankel matrix of order L from a given matrix of size TxM,
    where M is the number of features and T is the batch size.
    Note that we need L <= T.

    :param data:  A matrix of data (size TxM). 
                  T is the batch size and M is the number of features
    :param order: The order of the Hankel matrix (L)
    :return:      The Hankel matrix of type np.ndarray
    """
    data = np.array(data)
    
    assert len(data.shape) == 2, "Data needs to be a matrix"

    T, M = data.shape
    assert T >= order and order > 0, "The number of data points needs to be larger than the order"

    H = np.zeros((order * M, (T - order + 1)))
    for idx in range (T - order + 1):
        H[:, idx] = data[idx:idx+order, :].flatten()
    return H

def split_data(data: Data, Tini: int, horizon: int) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Utility function used to split the data into past data and future data.
    Constructs the Hankel matrix for the input/output data, and uses the first
    Tini rows to create the past data, and the last 'horizon' rows to create
    the future data.
    For more info check eq. (4) in https://arxiv.org/pdf/1811.05890.pdf

    :param data:    A tuple of input/output data. Data should have shape TxM
                    where T is the batch size and M is the number of features
    :param Tini:    number of samples needed to estimate initial conditions
    :param horizon: horizon
    :return:        Returns Up,Uf,Yp,Yf (see eq. (4) of the original DeePC paper)
    """
    assert Tini >= 1, "Tini cannot be lower than 1"
    assert horizon >= 1, "Horizon cannot be lower than 1"

    Mu, My = data.u.shape[1], data.y.shape[1]
    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)

    Up, Uf = Hu[:Tini * Mu], Hu[-horizon * Mu:]
    Yp, Yf = Hy[:Tini * My], Hy[-horizon * My:]
    
    return Up, Uf, Yp, Yf

def low_rank_matrix_approximation(
        X:NDArray[np.float64],
        explained_var: Optional[float] = 0.9,
        rank: Optional[int] = None,
        SVD: Optional[Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = None,
        **svd_kwargs) -> NDArray[np.float64]:
    """
    Computes an low-rank approximation of a matrix

    Adapted from https://gist.github.com/thearn/5424219

    :param X:               matrix to approximate
    :param explained_var:   Value in (0,1] used to compute the rank. Describes how close 
                            the low rank matrix is to the original matrix (in terms of the
                            singular values). Default value: 0.9
    :param rank:            rank order. To be used if you want to approximate the matrix by a specific
                            rank. By default is None. If different than None, then it will override the
                            explained_var parameter.
    :param SVD:             If not none, it should be the SVD decomposition (see numpy.linalg.svd) of X
    :param **svd_kwargs:    additional parameters to be passed to numpy.linalg.svd
    :return: the low rank approximation of X
    """
    assert len(X.shape) == 2, "X must be a matrix"
    assert explained_var is None and isinstance(rank, int) or isinstance(explained_var, float), \
        "You need to specify explained_var or rank!"
    assert explained_var is None or explained_var <= 1. and explained_var > 0, \
        "explained_var must be in (0,1]"
    assert rank is None or (rank >= 1 and rank <= min(X.shape[0], X.shape[1])), \
        "Rank cannot be lower than 1 or greater than min(num_rows, num_cols)"
    assert SVD is None or len(SVD) == 3, "SVD must be a tuple of 3 elements"

    u, s, v = np.linalg.svd(X, **svd_kwargs) if not SVD else SVD

    if rank is None:
        s_squared = np.power(s, 2)
        total_var = np.sum(s_squared)
        z = np.cumsum(s_squared) / total_var
        rank = np.argmax(np.logical_or(z > explained_var, np.isclose(z, explained_var)))

    X_low = np.zeros_like(X)

    for i in range(rank):
        X_low += s[i] * np.outer(u[:,i], v[i])
    return X_low

