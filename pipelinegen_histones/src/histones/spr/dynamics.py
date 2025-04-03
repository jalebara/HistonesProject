import numpy as np
import scipy
from typing import Callable, List, Tuple
from functools import partial
from multiprocessing import Process, Queue
from scipy.optimize import differential_evolution
from lmfit import Parameters


class ODEModel:
    def __init__(self, params: Parameters, ivp_method: str = "LSODA"):
        self.ivp_method = ivp_method
        self.params = params

    @property
    def converted_parameters(self):
        """Returns the parameters of the model converted to the correct units"""
        raise NotImplementedError

    def _ode(self, t: np.ndarray, y: np.ndarray, params: Parameters):
        """Returns the derivative of the state variables at the given time points"""
        raise NotImplementedError

    def _ode_solution(self, t: np.ndarray, y0: np.ndarray):
        sol = scipy.integrate.solve_ivp(
            self._ode,
            (t[0], t[-1]),
            y0=y0,
            t_eval=t,
            args=(self.params,),
        )
        return sol

    def __call__(self, t: np.ndarray, y0: np.ndarray, **kwargs):
        concentration = kwargs.get("concentration", None)
        if concentration is not None:
            self.params["Ct"].value = concentration
        return self.evaluate(t, y0)

    def evaluate(self, t: np.ndarray, y0: np.ndarray):
        """Returns the solution of the ODE at the given time points for the variable of interest"""
        sol = self._ode_solution(t, y0)
        return sol.y[0]


class TwoCompartmentModel(ODEModel):
    def __init__(self, params: Parameters, ivp_method: str = "LSODA"):
        """Initializes a Two Compartment Model

        Expected Parameters:
            - Ka is a free parameter
            - Kd is a free parameter
            - Rt is a free parameter
            - Km is a free parameter
            - Ct is fixed
            - vc is fixed
            - D is fixed
            - H is fixed
            - L is fixed
            - S is fixed

        Args:
            params (Parameters): parameters of the model
            ivp_method (str, optional): The method used to solve the IVP. Defaults to "LSODA".
        """
        super().__init__(params, ivp_method)

    def _ode(
        self, t: np.ndarray, y: np.ndarray, params: Parameters
    ) -> List[np.ndarray]:
        """Definition of the coupled ODEs for the two compartment model as described by Myszka et. al in the

        Assumptions:
        The units of the parameters are as follows:
            - Ka is in M^-1 s^-1
            - Kd is in s^-1
            - Rt and B are in RU
            - C and Ct are in M
            - tKm is in 1/s
            - S is in cm^2
            - Vi is in cm^3
            - vc is in cm/s
            - H, L are in cm

        The ODE is defined as follows:
            Cdot = S * (-Ka * C * (Rt - tB) + Kd * tB + tKm * (Ct - C))
            Bdot = Ka * C * (Rt - tB) - Kd * tB

        where tB = B/h_i, tKm = Km/h_i, and h_i is the height of the inner compartment.
        For mathematical convenience, we set h_i = 1 RU/M.

        B is the bound receptor concentration, C is the inner compartment concentration,
        Ct is the concentration of analyte in the bulk, Rt is the total receptor concentration,
        Ka is the association constant, Kd is the dissociation constant, Km is the mass transport
        constant, S is the surface area of the sensing cell, Vi is the volume of the flow cell,
        H is the height of the flow cell, L is the length of the flow cell, and vc is the maximum
        fluid velocity.


        Args:
            t (np.ndarray): time array
            y (np.ndarray): state variables
            params (Parameters): parameters of the model

        Returns:
            List[np.ndarray]: derivatives of the state variables C and B
        """
        # get arguments from params dict
        Ka = params["Ka"]
        Kd = params["Kd"]
        Rt = params["Rt"]
        Ct = params["Ct"]
        S = params["S"]
        #Vi = params["Vi"]
        # H = params["H"]
        # L = params["L"]
        # D = params["D"]
        vc = params["vc"]
        tKm = params["Km"]
        C, tB = y
        # Km = 1.282 * np.power((vc * D**2)/ (H * L), 1/3) # Mass Transport Constant
        Cdot = S * (-Ka * C * (Rt - tB) + Kd * tB + tKm * (Ct - C))
        Bdot = Ka * C * (Rt - tB) - Kd * tB
        return [Cdot, Bdot]


# Lorenz System
def lorenz_attractor(t: np.ndarray, d: np.ndarray, s: float, r: float, b: float):
    x, y, z = d
    xdot = s * (y - x)
    ydot = r * x - y - x * z
    zdot = x * y - b * z
    return [xdot, ydot, zdot]


# Langmuir Integrated Rate Equation
def integrated_langmuir_association(
    t: np.ndarray,
    Req: float,
    ka: float,
    kd: float,
    C: float,
) -> np.ndarray:
    """provides the integrated langmuir equation for a given time t

    Args:
        t (np.ndarray): time
        C (float): concentration
        Req (float): equilibrium constant
        ka (float): association constant
        kd (float): dissociation constant

    Returns:
        np.ndarray: Response
    """
    return Req * (1 - np.exp(-(ka * C + kd) * t))


# Langmuir Model
def langmuir_121(
    t: np.ndarray,
    d: np.ndarray,
    params: dict,
):
    """A Langmuir model with 1:1 binding"""
    L, LA = d
    ka = params["Ka"]
    kd = params["Kd"]
    Ct = params["Ct"]
    Ldot = -(ka * L * Ct - kd * LA)
    LAdot = (
        ka * L * Ct - kd * LA
    )  # equivalent to Ndot in Jin and Bdot in two compartment
    return [Ldot, LAdot]


# Two Compartment Model as an ODE
def two_compartment(
    t: np.ndarray,
    d: np.ndarray,
    params: dict,
):
    """A Two Compartment model as described by Myszka et. al in the
    1998 paper Extending the Range of Rate Constants Available form BIACORE:
    Interpreting Mass Transport Influenced Binding Data

    Args:
        d (np.ndarray): an array containing C and B, the inner compartment concentration and the bound receptor concentration.
        t (np.ndarray): time array
        S (float): Surface Area of sensing cell
        Ct (float): Injection Concentration of Analyte by time
        Ka (float): association constant
        Kd (float): dissociation constant
        Rt (float): total receptor concentration
        Vi (float): volume of flow cell
        H (float): height of flow cell
        L (float): length of flow cell
        D (float): Analyte Diffusion Coefficient
        vc (float): Maximum fluid velocity

    Returns:
        list: A list of the derivatives of the state variables C and B
    """
    # get arguments from params dict
    Ka = params["Ka"]
    Kd = params["Kd"]
    Rt = params["Rt"]
    Ct = params["Ct"]
    S = params["S"]
    Vi = params["Vi"]
    # H = params["H"]
    # L = params["L"]
    # D = params["D"]
    # vc = params["vc"]
    Km = params["Km"]
    C, B = d
    # Km = 1.282 * np.power((vc * D**2)/ (H * L), 1/3) # Mass Transport Constant
    Cdot = -Ka * C * (Rt - B) + Kd * B + Km * (Ct - C)
    Bdot = Ka * C * (Rt - B) - Kd * B
    return [Cdot, Bdot]


# Jin Models
def jin(
    t: float,
    d: float,
    Ka: float,
    Kd: float,
    L: float,
    H: float,
    W: float,
    D: float,
    Q: float,
    Ct: Callable,
    N0: float,
):
    """Implements the Jin Binding model as described in Jin et. al's
    General model for mass transport to planar and nanowire biosensor surfaces
    A planar cell is assumed

    Args:
        t (float): time
        d (float): contains N current density of binded sites
        L (float): Length of flow cell
        H (float): Height of flow cell
        W (float): width of flow cell
        D (float): Diffusion coefficient
        Q (float): Flow Rate
        p0 (Callable): Bulk concentration of analyte by time
        Ka (float): association constant
        Kd (float): dissociation constant
        N0 (float): density of receptor sites

    Returns:
        _type_: _description_
    """
    N = d
    Peh = Q / (D * W)  # Peclet number
    Pes = 6 * (L**2 / H**2) * Peh
    delch = 1 / Peh
    delf = L / (np.power(Pes, 1 / 3))
    deld = lambda t: np.sqrt(2 * D * t)
    deleff = lambda t: 1 / (1 / deld(t) + 1 / (delch + delf))
    Cd = lambda t: D / deleff(t)

    Ndot = (Ka * Ct(t) * (N0 - N) - Kd * N) / (1 + (Ka * (N0 - N)) / (Cd(t)))
    return Ndot


def build_residuals_function(
    rate_equation,
    C,
    ivp_guess: tuple = (0, 0),
    sol_index: int = 1,
    **rate_kwargs,
) -> Callable:
    """For a list of concentrations and a set of fixed parameters, return a function
    that calculates the residuals of the numerically integrated
    rate equation and the experimental data. The function returned by this function
    can be used by scipy.optimize.least_squares to find the best fit

    Args:
        rate_equation (Callable): The rate equation to be integrated
        time (np.ndarray): The time arrays of the experimental data
        C (np.ndarray): The concentrations of the experimental data
        **rate_kwargs (dict): The fixed parameters of the rate equation
    Returns:
        Callable: A function that takes the experimental data and returns the residuals
    """
    ## for each C, make a partial function that takes t and returns the rate equation
    partials = []
    for c in C:
        rate_equation = partial(rate_equation, Ct=c, **rate_kwargs)
        partials.append(rate_equation)

    def integrated_residuals(coeffs, time, data) -> np.ndarray:
        """Integrates the rate equation and returns the residuals of the experimental data and the integrated rate equation

        Args:
            yhat (np.ndarray): The experimental data
            *args: The arguments to be passed to the rate equation. These are the parameters to be optimized
        Returns:
            np.ndarray: array of residuals
        """
        residuals = []
        alpha = coeffs[0]
        params = coeffs[1:]
        processes = []
        queue = Queue()

        def multiprocessing_solver_wrapper(p, t, y0, args, i, q):
            result = scipy.integrate.solve_ivp(
                p, (t[0], t.max()), y0=y0, t_eval=t, args=args
            )
            q.put((i, result))
            return

        for i, p in enumerate(partials):
            # integrate the rate equation
            # the first arg is a scalar multiple of the integrated rate equation
            # the second and third args are ka and kd, respectively
            t = time[i]
            processes.append(
                Process(
                    target=multiprocessing_solver_wrapper,
                    args=(p, t, ivp_guess, params, i, queue),
                )
            )
            processes[-1].start()
            # solved = scipy.integrate.solve_ivp(
            #    p, (t[0], t[-1]), y0=ivp_guess, t_eval=t, args=params
            # )
        # calculate the residuals
        for _ in range(len(processes)):
            i, solved = queue.get()
            processes[i].join()
            d = data[i]
            residuals.append(d - alpha * solved.y[sol_index, :])
        return np.concatenate(residuals).flatten()

    return integrated_residuals
