import numpy as np
from typing import NamedTuple, Tuple

class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param y: output data
    """
    u: np.ndarray
    y: np.ndarray


def create_hankel_matrix(data: np.ndarray, order: int) -> np.ndarray:
    """
    Create an Hankel matrix of order L from a given matrix of size TxM,
    where M is the number of features and T is the batch size.
    Note that we need L <= T.

    :param data:    A matrix of data (size TxM). 
                    T is the batch size and M is the number of features
    :param order:   the order of the Hankel matrix (L)
    :return:        The Hankel matrix of type np.ndarray
    """
    data = np.array(data)
    T = len(data)
    assert len(data.shape) == 2, "Data needs to be a matrix"
    assert T >= order and order > 0, "The number of data points needs to be larger than the order"
    M = data.shape[1]

    H = np.zeros((order, (T - order + 1)*M))
    for idx in range (T - order + 1):
        H[:, idx*M : (idx + 1)*M] = data[idx:idx+order, :]
    return H

def split_data(data: Data, Tini: int, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    Hu = create_hankel_matrix(data.u, Tini + horizon)
    Hy = create_hankel_matrix(data.y, Tini + horizon)

    Up, Uf = Hu[:Tini], Hu[-horizon:]
    Yp, Yf = Hy[:Tini], Hy[-horizon:]
    
    return Up, Uf, Yp, Yf
