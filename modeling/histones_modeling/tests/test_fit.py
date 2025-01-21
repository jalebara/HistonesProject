import numpy as np
import pytest
from dynamics import Fitter, integrated_langmuir_association
from lmfit import Parameters
import pytest


def test_fit():
    # Create a dummy residual function for testing
    def residual_func(params, t, y0):
        return np.sum((params["param"] * t - y0) ** 2)

    # Create some dummy data for testing
    t = np.array([1, 2, 3, 4, 5])
    y0 = np.array([2, 4, 6, 8, 10])  # y=2x

    # Create an instance of the Fitter class
    params = Parameters()
    params.add("param", value=1)
    fitter = Fitter(residual_func, params)

    # Call the fit method and check the result
    result = fitter.fit(t, y0)
    assert np.allclose(result, np.array([0, 0, 0, 0, 0]))


@pytest.mark.parametrize(
    "Rt, ka, kd, Ct",
    [
        (30, 1.62e5, 9.15e-4, 50e-9),
        (30, 5.62e5, 2.25e-3, 50e-9),
        (1e5, 1e4, 1e-6, 1e-7),
    ],
)
def test_integrated_langmuir_fitting(Rt, ka, kd, Ct):
    # generate some data
    t = np.linspace(0, 1000, 10000)
    # True param values
    params_true = Parameters()
    params_true.add("Rt", value=Rt)
    params_true.add("ka", value=ka)
    params_true.add("kd", value=kd)
    params_true.add("Ct", value=Ct, vary=False)

    # simulate data
    y0 = integrated_langmuir_association(t, params_true)
    params = Parameters()
    params.add("Rt", value=1, min=1)
    params.add("ka", value=100, min=1e-1)
    params.add("kd", value=1e-1, max=1e-1)
    params.add("Ct", value=Ct, vary=False)

    def residuals(params, t, y0):
        return y0 - integrated_langmuir_association(t, params)

    fitter = Fitter(residuals, params, method="least_squares")
    result = fitter.fit(t, y0)
    yhat = integrated_langmuir_association(t, result.params)
    assert np.allclose(y0, yhat, rtol=1e-1)
    # assert that the fitted parameters are close to the true parameters
    assert np.allclose(result.params["Rt"].value, Rt, rtol=0.1)
    assert np.allclose(result.params["ka"].value, ka, rtol=0.1)
    assert np.allclose(result.params["kd"].value, kd, rtol=0.1)
