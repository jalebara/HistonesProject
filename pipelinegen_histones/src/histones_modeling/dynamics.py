import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
from lmfit import Parameters, minimize, report_fit

from typing import Callable, List, Tuple


## Electrodynamics


def MOSFET_linear_region(vth: float, params: Parameters):
    """
    Linear region of a MOSFET
    """
    mu = params["mu"].value
    cox = params["cox"].value
    w = params["w"].value
    l = params["l"].value
    vref = params["vref"].value
    vds = params["vds"].value

    return mu * cox * w / l * ((vref - vth) * vds - 0.5 * vds**2)


class Fitter:
    def __init__(
        self, residual_func: Callable, params: Parameters, method: str = "leastsq"
    ):
        self.f = residual_func
        self.params = params
        self.method = method

    def fit(self, t: np.ndarray, y0: np.ndarray) -> np.ndarray:
        self.result = minimize(self.f, self.params, args=(t, y0), method=self.method)
        report_fit(self.result)
        return self.result


# test function
def integrated_langmuir_association(t: np.ndarray, params: Parameters) -> np.ndarray:
    """
    Integrated Langmuir association model
    """
    Rt = params["Rt"].value
    ka = params["ka"].value
    kd = params["kd"].value
    Ct = params["Ct"].value

    return Rt * (1 - np.exp(-(ka * Ct + kd) * t))


class ODE:
    def __init__(
        self,
        y0: np.ndarray,
        params: Parameters,
        f: Callable,
        jac: Callable = None,
    ):
        self.params = params
        self.f = f
        self.jac = jac
        self.y0 = y0

    def solve(self, t) -> np.ndarray:
        self.sol = scipy.integrate.solve_ivp(
            fun=self.f,
            t_span=(t[0], t[-1]),
            y0=self.y0,
            method="BDF",
            jac=self.jac,
            t_eval=self.t,
            args=(self.params,),
        )
        return self.sol.y

    def __call__(self, t: np.ndarray, y0: np.ndarray, params: Parameters) -> np.ndarray:
        self.t = t
        self.y0 = y0
        self.params = params
        return self.solve()


def langmuir121(t: np.ndarray, d: np.ndarray, params: Parameters):
    """
    Langmuir 1:1 model
    """
    L, LA = d
    ka = params["ka"].value
    kd = params["kd"].value
    Ct = params["Ct"].value
    Ldot = -(ka * L * Ct - kd * LA)
    LAdot = ka * L * Ct - kd * LA

    return np.array([Ldot, LAdot])
