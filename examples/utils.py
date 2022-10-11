import numpy as np
import scipy.signal as scipysig
from typing import Optional
from pydeepc import Data

class System(object):
    """
    Represents a dynamical system that can be simulated
    """
    def __init__(self, sys: scipysig.StateSpace, x0: Optional[np.ndarray] = None):
        """
        :param sys: a linear system
        :param x0: initial state
        """
        assert x0 is None or sys.A.shape[0] == len(x0), 'Invalid initial condition'
        self.sys = sys
        self.x0 = x0 if x0 is not None else np.zeros(sys.A.shape[0])
        self.u = None
        self.y = None

    def apply_input(self, u: np.ndarray, noise_std: float = 0.5) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        T = len(u)
        if T > 1:
            # If u is a signal of length > 1 use dlsim for quicker computation
            t, y, x0 = scipysig.dlsim(self.sys, u, t = np.arange(T) * self.sys.dt, x0 = self.x0)
            self.x0 = x0[-1]
        else:
            y = self.sys.C @ self.x0
            self.x0 = self.sys.A @ self.x0.flatten() + self.sys.B @ u.flatten()

        y = y + noise_std * np.random.normal(size = T).reshape(T, 1)

        self.u = np.vstack([self.u, u]) if self.u is not None else u
        self.y = np.vstack([self.y, y]) if self.y is not None else y
        return Data(u, y)

    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])

    def get_all_samples(self) -> Data:
        """
        Returns all samples
        """
        return Data(self.u, self.y)

    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        self.x0 = x0 if x0 is not None else np.zeros(self.sys.A.shape[0])



