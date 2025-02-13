import numpy as np
from types import SimpleNamespace

# Enumerate Known parameters for SPR machine
# Physical Constants
NA = 6.02214e23  # Avodgadro's number
R = 8.31446261815324  # Gas Constant in SI units
T = 293  # standard room temperature in Kelvin
eta = 10e-3  # Viscosity of water at room temperature in Pa*s
# solver vars
t_range_a = [0, 180]
t_range_d = [0, 600]

# Sampling rate of 10Hz
t_eval_a = np.linspace(0, t_range_a[-1], t_range_a[-1] * 10)
t_eval_d = np.linspace(0, t_range_d[-1], t_range_d[-1] * 10)

# Physical Properties Reconditioned for numerical stability
molec_mass = 11.0  # kDa
molec_length = 103  # estimated in nm
# molec_mass = molec_mass * 1000**2 * 1.66033e-27  # grams
# convert to 1/(nmol/m^3 *s)
ka = 5.62e5  # 1/(M*s) M is molarity (mol/dm^3) s is seconds
kd = 2.25e-3  # 1/s s is seconds
H = 4e-3  # cm
L = 0.24  # cm
# From https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5418478/
W = 5e-2  # cm to
S = 0.015  # cm^2
L = S / W
V = S * H
# convert uL/min to uL/s to cm^3/s
vc = 30 / 60 * 0.001  # cm^3/s
# diffusion coefficeint is still unknown
# Can estimate from the Stokes-Einstein equation but that
# requires that I model the geometry of the molecule
D = 0.1e-6  # cm^2/s
# D = R *T/NA * 1 / (6 * np.pi  * eta * molec_length/2) # Stokes-Einstein
Km = 1.282 * np.power(
    (vc * D**2) / (H * L), 1 / 3
)  # Mass Transport Constant in Two Compartment model
# an RU is 1pg of material --> 10e-3 degree shift
# convert 1pg to 1kDa
ImmobilizationRU = (
    1000 / 1e12 / 1.66033e-27 / 1000
)  # RU maybe there's a way to estimate a theoretical quantity for this
# kDa to nmoles/cm^3
Rt = (ImmobilizationRU / molec_mass / NA * 1e9) / S

# convert everything to the same units
biacoreT2000 = SimpleNamespace(
    molec_mass=molec_mass,
    molec_length=molec_length,
    ka=ka,
    kd=kd,
    H=H,
    L=L,
    W=W,
    S=S,
    V=V,
    vc=vc,
    D=D,
    Km=Km,
    ImmobilizationRU=ImmobilizationRU,
    Rt=Rt,
)
