from ray import tune
from lmfit import Parameters, Parameter
import numpy as np
import tsfel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple, Dict, Any
from numpy.random import normal
import matplotlib.pyplot as plt

# custom imports
from spr.dynamics import TwoCompartmentModel
from spr.biacoreT2000 import biacoreT2000

import numpy as np
from scipy.special import lambertw

# SPR Parameters
NA = 6.02214e23  # Avodgadro's number
R = 8.31446261815324  # Gas Constant in SI units
T = 293  # standard room temperature in Kelvin
eta = 10e-3  # Viscosity of water at room temperature in Pa*s
# solver vars``
t_range_a = [0, 180]
t_range_d = [0, 600]
molec_mass = 11.0  # kDa
molec_length = 103  # estimated in nm
H = 4e-3  # cm
L = 0.24  # cm
# From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5418478/
W = 5e-2  # cm to
S = L*W
vc = 30 / 60 * 0.001  # cm^3/s
# Estimate Diffusion Coefficient
D = 0.1e-6  # cm^2/s
ImmobilizationRU = 1000 / 1e12 / 1.66033e-27 / 1000
N0 = (ImmobilizationRU / molec_mass / NA * 1e9) / S * 1e-9


def einstein_sutherland(molec_weight, eta=0.001, fof0=1.2, vbar=7e-4):
    """Implements the Einstein Sutherland equation for estimating the diffusion coefficient of a molecule in a fluid.

    Args:
        molec_weight (float): The molecular weight of the molecule in kDa.
        eta (float, optional): The viscosity of the fluid in Pa*s. Defaults to 0.001.
        flf0 (float, optional): The ratio of the fluid's viscosity to the viscosity of water. Defaults to 1.2.
        vbar (float, optional): The specific density of the analyte in m^3/kg. Defaults to 7e-4.

    Returns:
        float: The estimated diffusion coefficient in cm^2/s.

    """
    k = 1.38064852e-23  # Boltzmann constant in J/K
    T = 293  # room temperature in K
    NA = 6.02214076e23  # Avogadro's number
    res = (k * T) / (
        6
        * np.pi
        * np.power(3 * molec_weight * vbar / (4 * np.pi * NA), 1 / 3)
        * eta
        * fof0
    )
    # convert to cm^2/s
    res *= 1e4
    return res


def scaled_response(t, tc, C, F, l, l1, l2, h, w, N0, D, ka, kd):
    """
    Reponse normalized (between 0 and 1) by the theoretical max. response. To get SPR response,
    multiply by Rmax. To get surface density, multiply by N0.

    Based on L. L. H. Christensen, “Theoretical Analysis of Protein Concentration Determination Using
    Biosensor Technology under Conditions of Partial Mass Transport Limitation,” Analytical Biochemistry,
    vol. 249, no. 2, pp. 153–164, Jul. 1997, doi: 10.1006/abio.1997.2182.

    Inputs:
        t: numpy array of time values in seconds
        tc: cutoff time between association and dissociation, s
        C: bulk concentration of analyte, M
        F: flow rate, cm^3/s
        l: cell length, cm
        l1: length to the start of the detection area from the inlet of the flow cell, cm
        l2: length to the end of the detection area from the inlet of the flow cell, cm
        h: cell height, cm
        w: cell width, cm
        N0: total amount of ligand on the surface (i.e., available binding sites) at time 0, #/cm^2
        D: diffusion constant, cm^2/s
        ka: association constant, M^-1 s^-1
        kd: dissociation constant, s^-1
    """

    ta = t[t <= tc]
    td = t[t > tc]

    Ckc = 1.47 * (1 - ((l1 / l2) ** (2 / 3))) / (1 - l1 / l2)  # unitless
    kc = Ckc * ((D**2) * F / (h**2 * w * l2)) ** (1 / 3)  # cm/s

    alpha = kc * C / (kd * N0)
    beta = kc / (ka * N0)

    # association
    K1 = alpha / (alpha + beta)
    K2 = alpha / (beta * (1 + alpha + beta))
    K3 = ((alpha + beta) ** 2) / (beta * (1 + alpha + beta))
    Rta = np.real(K1 * (1 - (1 / K2) * lambertw(K2 * np.exp(K2 - K3 * kd * ta))))

    # dissociation
    Rtc = np.real(Rta[-1])
    Rtd = np.real(
        -(beta + 1)
        * lambertw(
            -Rtc * np.exp(-(beta * kd * (td - tc) + Rtc) / (beta + 1)) / (beta + 1)
        )
    )

    return np.concatenate((Rta, Rtd))


class PartialTransportObjective:
    def __init__(self, randomized_params: List[Tuple[str, float, float]]):
        """
        Assuming a normal distribution for each parameter, the randomized_params
        are sampled at the start of the optimization process. The first element
        of each tuple is the parameter name, the second is the mean of the distribution,
        and the third is the standard deviation.
        """
        self.randomized_params = {}

    def _sample_parameter_distribution(
        self, param_specification: Dict[str, Tuple[float, float]]
    ):
        samples = {
            param: np.random.normal(mean, std)
            for param, mean, std in param_specification
        }
        params = [Parameter(key, sampled) for key, sampled in samples.items()]
        return params

    def objective(self, controlled_params: Dict[str, Any]):
        dataset = []
        time_range = np.linspace(0, 1000, 2000)
        for i in range(100):
            params = self._sample_parameter_distribution(self.randomized_params)
            params.update(controlled_params)
            concentration_space = np.linspace(1e-9, 1e-3, 100)
            for c in concentration_space:
                sample = scaled_response(
                    time_range,
                    600,
                    c,
                    params["F"],
                    params["l"],
                    params["l1"],
                    params["l2"],
                    params["h"],
                    params["w"],
                    params["N0"],
                    params["D"],
                    normal(*params["ka"]),
                    normal(*params["kd"]),
                )
                dataset.append(sample)
        dataset = np.asarray(dataset)

        # now we can compute the learning objective
        labels = np.zeros((dataset.shape[0] * 2, 1))
        labels[dataset.shape[0] :] = 1
        dataset = np.concatenate((dataset, dataset), axis=0)

        # extract features from the dataset
        cfg = tsfel.get_features_by_domain("temporal")

        features = tsfel.time_series_features_extractor(cfg, list(dataset), fs=10)
        # Remove low variance features
        sel = VarianceThreshold()
        features = sel.fit_transform(features)
        # Select the best features
        features = SelectKBest(f_classif, k=10).fit_transform(features, labels)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Train the classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)


class TwoCompartmentModelObjective:
    def __init__(
        self,
        randomized_params: list[tuple[str, float, float]],
        analyte_a_randomized: list[tuple[str, float, float]],
        analyte_b_randomized: list[tuple[str, float, float]],
    ):
        """
        Assuming a normal distribution for each parameter, the randomized_params
        are sampled at the start of the optimization process. The first element
        of each tuple is the parameter name, the second is the mean of the distribution,
        and the third is the standard deviation.
        """
        self.randomized_params = randomized_params
        self.analyte_a_randomized = analyte_a_randomized
        self.analyte_b_randomized = analyte_b_randomized

    def _sample_parameter_distribution(
        self, param_specification: list[tuple[str, float, float]]
    ):
        samples = {
            param: np.random.normal(mean, std)
            for param, mean, std in param_specification
        }
        params = [Parameter(key, sampled) for key, sampled in samples.items()]
        return params

    def objective(self, params: list[Parameter]):
        analyte_a_params_base = self.randomized_params.copy()
        analyte_a_params_base.extend(self.analyte_a_randomized)
        analyte_b_params_base = self.randomized_params.copy()
        analyte_b_params_base.extend(self.analyte_b_randomized)

        analyte_a_dataset = []
        analyte_b_dataset = []
        time_range = np.linspace(0, 800, 1000)
        for i in range(100):
            analyte_a_params = self._sample_parameter_distribution(
                analyte_a_params_base
            )
            analyte_b_params = self._sample_parameter_distribution(
                analyte_b_params_base
            )

            a_params = Parameters()
            a_params.add_many(*analyte_a_params)
            a_params.add_many(*params)

            b_params = Parameters()
            b_params.add_many(*analyte_b_params)
            b_params.add_many(*params)

            analyte_a_model = TwoCompartmentModel(a_params)
            analyte_b_model = TwoCompartmentModel(b_params)

            concentration_space = np.linspace(1e-9, 1e-3, 100)
            for c in concentration_space:
                analyte_a_sample = analyte_a_model(time_range, (0, 0), concentration=c)
                analyte_b_sample = analyte_b_model(time_range, (0, 0), concentration=c)
                analyte_a_dataset.append(analyte_a_sample)
                analyte_b_dataset.append(analyte_b_sample)
        analyte_a_dataset = np.asarray(analyte_a_dataset)
        analyte_b_dataset = np.asarray(analyte_b_dataset)

        # now we can compute the learning objective
        labels = np.zeros((analyte_a_dataset.shape[0] * 2, 1))
        labels[analyte_a_dataset.shape[0] :] = 1
        dataset = np.concatenate((analyte_a_dataset, analyte_b_dataset), axis=0)

        # extract features from the dataset
        cfg = tsfel.get_features_by_domain("temporal")

        features = tsfel.time_series_features_extractor(cfg, list(dataset), fs=10)
        # Remove low variance features
        sel = VarianceThreshold()
        features = sel.fit_transform(features)
        # Select the best features
        features = SelectKBest(f_classif, k=10).fit_transform(features, labels)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Train the classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)


def tune_partial_transport():
    hh4_molec_weight = 12.4 # kDa
    calf_h3_molec_weight = 15.3 # kDa
    hh3_molec_weight = 17. # kDa

    common_params = {
        "l1": 0.03,
        "l2": 0.2,
        "N0": N0,
    }

    search_space = {
        "F": tune.loguniform(1e-3, 1e-1),
        "h": tune.uniform(0.001, 0.01),
        "l": tune.uniform(0.23, 0.5),
        "w": tune.uniform(0.01, 0.1),
    }

    analyte_params = [
        {
            "ka": (5.62e5, 0.05e5),
            "kd": (9.15e-4, 0.06e-4),
            "D": einstein_sutherland(hh4_molec_weight),
        },
        {
            "ka": (1.62e5, 0.03e5),
            "kd": (9.15e-4, 0.06e-4),
            "D": einstein_sutherland(calf_h3_molec_weight),
        }
    ]

    def objective(params):
        """ Expects a dictionary of parameters

         The dictionary should contain the following keys
            - F: flow rate in cm^3/s
            - h: cell height in cm
            - l: cell length in cm
            - w: cell width in cm
            - analyte_params: a dictionary containing the following keys
                - ka: association constant and standard deviation in M^-1 s^-1
                - kd: dissociation constant and standard deviation in s^-1
                - D: diffusion constant in cm^2/s
            - l1: length to the start of the detection area from the inlet of the flow cell, cm
            - l2: length to the end of the detection area from the inlet of the flow cell, cm
            - N0: total amount of ligand on the surface (i.e., available binding sites) at time 0,
        Args:
            params (dict): A dictionary of parameters
        Returns:
            float: The objective value (i.e. accuracy)
        """
        # generate configs for each analyte
        analyte_params = params.pop("analyte_params")
        dataset = []
        targets = []
        time_range = np.linspace(0, 600, 6000)
        concentration_space = np.linspace(1e-9, 200e-9, 100)
        for c in concentration_space:
            for trg, analyte in enumerate(analyte_params):
                sample = scaled_response(
                    time_range,
                    200,
                    c,
                    params["F"],
                    params["l"],
                    params["l1"],
                    params["l2"],
                    params["h"],
                    params["w"],
                    params["N0"],
                    analyte["D"],
                    normal(*analyte["ka"]),
                    normal(*analyte["kd"])
                )
                dataset.append(sample)
                targets.append(trg)

        dataset = np.asarray(dataset)
        targets = np.asarray(targets)

        # extract features from the dataset
        cfg = tsfel.get_features_by_domain("temporal")
        features = tsfel.time_series_features_extractor(cfg, list(dataset), fs=2, verbose=0)
        # Remove low variance features
        sel = VarianceThreshold()
        features = sel.fit_transform(features)
        # Select the best features
        features = SelectKBest(f_classif, k=10).fit_transform(features, targets)
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        # Normalize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Train the classifier
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return accuracy_score(y_test, y_pred)

    search_space = {**common_params, **search_space}
    search_space["analyte_params"] = analyte_params
    tune.run(
        objective,
        config=search_space,
        num_samples=100,
        resources_per_trial={"cpu": 3},
    )





def tune_objective(common_params, a_params, b_params, params: dict):
    objective = PartialTransportObjective(common_params)
    params = [Parameter(key, value) for key, value in params.items()]
    return objective.objective(params)


def main():
    common_params = [
        ("Rt", biacoreT2000.Rt, 0.1),
        ("Km", biacoreT2000.Km, 1e-6),
        ("D", biacoreT2000.D, 1e-6),
        ("Ct", biacoreT2000.vc, 1e-4),
    ]

    analyte_a_params = [
        ("Ka", biacoreT2000.ka, 10),
        ("Kd", biacoreT2000.kd, 1e-4),
    ]

    analyte_b_params = [
        ("Ka", biacoreT2000.ka * 2, 10),
        ("Kd", biacoreT2000.kd / 2, 1e-4),
    ]

    search_space = {
        "vc": tune.uniform(1e-6, 1e-3),
        "S": tune.uniform(0.001, 0.1),
    }

    # test
    print(
        tune_objective(
            common_params, analyte_a_params, analyte_b_params, {"vc": 1e-5, "S": 0.01}
        )
    )


if __name__ == "__main__":
    import ray
    ray.init()
    tune_partial_transport()
