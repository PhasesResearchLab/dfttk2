"""
Debye-Grüneisen module to calculate the vibrational contribution to the Helmholtz energy, entropy, and heat capacity.    
"""

# Standard library imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import constants
from scipy.special import bernoulli, gamma

# DFTTK imports
from dfttk import eos_fit
from dfttk.aggregate_extraction import extract_configuration_data
from dfttk.data_extraction import extract_average_mass
from dfttk.plotly_format import plot_format


# The definition of the Debye temperature is A * hbar/k_B * N/V) where A is
# A = (6 * pi^2)^1/3 * hbar/k_B = 2.977*10^-11 s*K
# converting to the of the eos_parameters,
# A = (hbar/kb) * (6*math.pi**2)**(1/3) * (1*10**-10 m/Å)**(1/2) * (1*10**12 (g/(ms^2))/GPa)**(1/2) / ((1.66054*10**-24 g/u)**(1/2))
# A = 231.0389521318254 K/(Å*GPa/u)^1/2
A = 231.0389521318254
BOLTZMANN_CONSTANT = constants.physical_constants["Boltzmann constant in eV/K"][0]


def gruneisen_parameter(bulk_modulus_prime: float, gruneisen_x: float) -> float:
    """Calculates the gruneisen parameter, gamma

    Args:
        bulk_modulus_prime: B_0', first derivative of the bulk modulus with respect to pressure
        gruneisen_x: Typical values include x = 2/3 (high temperature) and x = 1 (low temperature)

    Returns:
        gamma, the gruneisen parameter
    """
    return (1 + bulk_modulus_prime) / 2 - gruneisen_x


def debye_temperature(
    volume: float,
    eos_parameters: tuple[float],
    mass: float,
    gru_param: float,
    scaling_factor: float = 0.617,
) -> float:
    """Calculates the Debye temperature within the Debye-Gruneisen model.

    Args:
        volume: Volume of the cell
        eos_parameters: The parameters of the equation of state: (volume_0, energy_0, bulk_modulus, bulk_modulus_prime)
        mass: Total mass of the cell
        gru_param: gruneisen parameter
        scaling_factor: The scaling factor defaults to 0.617, as determined by Moruzzi et al. from their study on
        nonmagnetic cubic metals (https://doi.org/10.1103/PhysRevB.37.790).

    Returns:
        Debye temperature
    """

    s = scaling_factor
    volume_0 = eos_parameters[0]
    bulk_modulus = eos_parameters[2]

    return (
        s
        * A
        * volume_0 ** (1 / 6)
        * (bulk_modulus / mass) ** (1 / 2)
        * (volume_0 / volume) ** gru_param
    )


def debye_function(
    x_array: np.array, prec: float = 1e-12, nth_bernoulli: int = 100
) -> np.array:
    r"""Calculates the debye function with n = 3 using one of two series expansions. Valid for |x|<2𝜋 and n≥1.

    For -2pi < x < 0.7𝜋:
    D_3(x) = 1 - \frac{3}{8}x + 3 \sum_{k=1}^{\infty} \frac{B_{2k}}{(2k+3) \Gamma(2k+1)} x^{2k}
    Reference: Eq. (4) by Gonzalez et al. (https://doi.org/10.3390/math10101745)

    For x >= 0.7𝜋:
    D(x) = \frac{\pi^4}{5x^3} - 3 \sum_{k=1}^{\infty} \frac{1}{k} \left(1 + \frac{3}{kx} + \frac{6}{k^2x^2} + \frac{6}{k^3x^3}\right) e^{-kx}
    Reference: Eq. (17) by Khishchenko (https://doi.org/10.20948/MATHMONTIS-2020-49-8)

    Other reference:
    Abramowitz, M. and Stegun, I.A. eds., 1968. Handbook of mathematical functions with formulas, graphs, and mathematical tables (Vol. 55).

    Args:
        x_array: array of input values for the debye function
        prec: Precision. Terminates the series expansion when the absolute value of the term is less than prec.
        nth_bernoulli: Determines the nth Bernoulli number to calculate. A list of Bernoulli numbers is generated prior to calculating the series
        expansion. There should be no reason to change this value under normal circumstances.

    Raises:
        ValueError: If the precision is not between 0 and 1
        ValueError: If x < -2𝜋
        IndexError: If the bernoulli number at index 2k is not available. This indicates slow convergence of the Debye function series expansion.
        If you wish to calculate values for x < -𝜋, convergence may be slow and you may need to increase nth_bernoulli.

    Returns:
        np.array: The value of the debye function evaluated at each x in x_array
    """

    if not 0 < prec < 1:
        raise ValueError("The precision must be between 0 and 1")

    result = np.zeros_like(x_array)
    bern_list = bernoulli(nth_bernoulli)  # 2*k must be less than 100

    for i, x in enumerate(x_array):
        term = 1  # Ensures the while loop runs at least once
        k = 1

        if x >= 0.7 * np.pi:
            debye_value = np.pi**4 / (5 * x**3)
            while abs(term) > prec:
                term = -3 * (
                    1
                    / k
                    * (1 + 3 / (k * x) + 6 / (k**2 * x**2) + 6 / (k**3 * x**3))
                    * np.exp(-k * x)
                )
                debye_value += term
                k += 1
            result[i] = debye_value

        elif -2 * np.pi < x < 0.7 * np.pi:
            debye_value = 1 - 3 / 8 * x
            while abs(term) > prec:
                try:
                    term = 3 * (
                        bern_list[2 * k]
                        / ((2 * k + 3) * gamma(2 * k + 1))
                        * x ** (2 * k)
                    )
                except IndexError:
                    raise IndexError(
                        f"IndexError: the bernoulli number at index {2*k} is not available. This indicates slow convergence of the Debye function series expansion. \
                            If you wish to calculate values for x < -𝜋, convergence may be slow and you may need to increase nth_bernoulli."
                    )
                debye_value += term
                k += 1
            result[i] = debye_value

        else:
            raise ValueError(
                "The debye function series expansions used are only valid for x > -2𝜋"
            )

    return result


def vibrational_energy(temperature: float, theta: float, number_of_atoms) -> float:
    """Evaluates the debye function at x = theta/temperature, then calculates the vibrational energy in eV.

    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        number_of_atoms: Number of atoms in the cell

    Returns:
        float: Vibrational energy in eV
    """

    zero_temp_mask = temperature == 0
    non_zero_temp_mask = temperature > 0

    e_vib = np.zeros_like(temperature)
    debye_value = np.zeros_like(temperature)

    e_vib[zero_temp_mask] = number_of_atoms * (9 / 8 * BOLTZMANN_CONSTANT * theta)

    debye_value[non_zero_temp_mask] = debye_function(
        theta / temperature[non_zero_temp_mask]
    )

    e_vib = (
        number_of_atoms
        * BOLTZMANN_CONSTANT
        * (
            3 * temperature[non_zero_temp_mask] * debye_value[non_zero_temp_mask]
            + 9 / 8 * theta
        )
    )
    return e_vib


def vibrational_entropy(temperature: float, theta: float, number_of_atoms) -> float:
    """Evaluates the debye function at x = theta/temperature, then calculates the vibrational entropy in eV/K.

    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        number_of_atoms: Number of atoms in the cell

    Returns:
        float: Vibrational entropy in eV/K

    """

    zero_temp_mask = temperature == 0
    non_zero_temp_mask = temperature > 0

    s_vib = np.zeros_like(temperature)
    x = np.zeros_like(temperature)
    debye_value = np.zeros_like(temperature)

    s_vib[zero_temp_mask] = 0

    x[non_zero_temp_mask] = theta / temperature[non_zero_temp_mask]
    debye_value[non_zero_temp_mask] = debye_function(x[non_zero_temp_mask])

    s_vib[non_zero_temp_mask] = (
        3
        * number_of_atoms
        * BOLTZMANN_CONSTANT
        * (
            4 / 3 * debye_value[non_zero_temp_mask]
            - np.log(1 - np.exp(-x[non_zero_temp_mask]))
        )
    )

    return s_vib


def vibrational_helmholtz_energy(
    temperature: np.ndarray | float, theta: float, number_of_atoms
) -> float:
    """Evaluates the debye function at x = theta/temperature, then calculates the vibrational Helmholtz energy in eV.

    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        number_of_atoms: Number of atoms in the cell

    Returns:
        float: Vibrational Helmholtz energy in eV
    """

    zero_temp_mask = temperature == 0
    non_zero_temp_mask = temperature > 0

    f_vib = np.zeros_like(temperature)
    x = np.zeros_like(temperature)
    debye_value = np.zeros_like(temperature)

    # Zero point energy
    f_vib[zero_temp_mask] = number_of_atoms * (9 / 8 * BOLTZMANN_CONSTANT * theta)

    x[non_zero_temp_mask] = theta / temperature[non_zero_temp_mask]
    debye_value[non_zero_temp_mask] = debye_function(x[non_zero_temp_mask])

    f_vib[non_zero_temp_mask] = number_of_atoms * (
        9 / 8 * BOLTZMANN_CONSTANT * theta
        + BOLTZMANN_CONSTANT
        * temperature[non_zero_temp_mask]
        * (
            3 * np.log(1 - np.exp(-x[non_zero_temp_mask]))
            - debye_value[non_zero_temp_mask]
        )
    )

    return f_vib


def vibrational_heat_capacity(
    temperature: np.ndarray | float, theta: float, number_of_atoms
) -> float:
    """Evaluates the debye function and its derivative at x = theta/temperature, then calculates the vibrational heat capacity in eV/K.
    The formula is taken from Eq. (13) from Khishchenko (https://doi.org/10.20948/MATHMONTIS-2020-49-8).

    Args:
        temperature : Temperature in Kelvin
        theta: Debye temperature in Kelvin
        number_of_atoms: Number of atoms in the cell

    Returns:
        float: Vibrational heat capacity in eV/K
    """

    zero_temp_mask = temperature == 0
    non_zero_temp_mask = temperature > 0

    cv_vib = np.zeros_like(temperature)
    x = np.zeros_like(temperature)
    debye_value = np.zeros_like(temperature)

    cv_vib[zero_temp_mask] = 0

    x[non_zero_temp_mask] = theta / temperature[non_zero_temp_mask]
    debye_value[non_zero_temp_mask] = debye_function(x[non_zero_temp_mask])

    cv_vib[non_zero_temp_mask] = (
        3
        * number_of_atoms
        * BOLTZMANN_CONSTANT
        * (
            4 * debye_value[non_zero_temp_mask]
            - 3 * x[non_zero_temp_mask] / (np.exp(x[non_zero_temp_mask]) - 1)
        )
    )
    return cv_vib


def plot_debye(
    debye_properties: pd.DataFrame,
    selected_temperatures_plot: np.array = None,
    selected_volumes: np.array = None,
    volume_decimals: int = 2,
    temperature_decimals: int = 0,
) -> tuple[go.Figure, go.Figure]:
    """Plots the vibrational properties (S,F,C_V) as a function of temperature and volume

    Args:
        temperatures: Array of temperatures
        volumes: Array of volumes
        number_of_atoms: Number of atoms in the cell
        y: Array of vibrational properties (S,F,C_V)
        y_label: Label for the y-axis ('S', 'F', 'C_V')
        selected_temperatures_plot: Array of selected temperatures curves to plot in the y vs volume plot. If None, 5 linearly spaced temperatures are selected.
        selected_volumes: Array of selected volumes curves to plot in the y vs temperature plot. If None, 5 linearly spaced volumes are selected.
        volume_decimals: Number of decimals to display for the volume in the plot
        temperature_decimals: Number of decimals to display for the temperature in the plot

    Returns:
        tuple[go.Figure, go.Figure]: Two plotly figures, one for the vibrational properties as a function of temperature and one for the vibrational properties
        as a function of volume
    """

    temperatures = debye_properties["temperatures"].values
    volumes = debye_properties["volume"].values[0]
    number_of_atoms = debye_properties["number_of_atoms"].values[0]

    f_vib = np.array(debye_properties["f_vib"].tolist()).T
    s_vib = np.array(debye_properties["s_vib"].tolist()).T
    cv_vib = np.array(debye_properties["cv_vib"].tolist()).T
    y_values = [f_vib, s_vib, cv_vib]
    y_labels = [
        f"F<sub>vib</sub> (eV/{number_of_atoms} atoms)",
        f"S<sub>vib</sub> (eV/K/{number_of_atoms} atoms)",
        f"C<sub>v,vib</sub> (eV/K/{number_of_atoms} atoms)",
    ]

    for y, y_label in zip(y_values, y_labels):
        s_t_fig = go.Figure()
        if selected_volumes is None:
            indices = np.linspace(0, len(volumes) - 1, 5, dtype=int)

        else:
            indices = []
            for v in selected_volumes:
                try:
                    indices.append(np.where(volumes == v)[0][0])
                except IndexError:
                    nearest_volume = volumes[np.argmin(np.abs(volumes - v))]
                    indices.append(np.where(volumes == nearest_volume)[0][0])

        for i, volume in enumerate(volumes):
            if i in indices:
                s_t_fig.add_trace(
                    go.Scatter(
                        x=temperatures,
                        y=y[i],
                        mode="lines",
                        name=f"{volume:.{volume_decimals}f} \u212B<sup>3</sup>",
                    )
                )
        plot_format(
            s_t_fig,
            "Temperature (K)",
            y_label,
        )
        s_t_fig.show()

    for y, y_label in zip(y_values, y_labels):
        s_v_fig = go.Figure()
        if selected_temperatures_plot is None:
            indices = np.linspace(0, len(temperatures) - 1, 5, dtype=int)
            selected_temperatures_plot = np.array([temperatures[j] for j in indices])
        else:
            indices = []
            for t in selected_temperatures_plot:
                try:
                    indices.append(np.where(temperatures == t)[0][0])
                except IndexError:
                    nearest_temperature = temperatures[
                        np.argmin(np.abs(temperatures - t))
                    ]
                    indices.append(np.where(temperatures == nearest_temperature)[0][0])
        for i in indices:
            s_v_fig.add_trace(
                go.Scatter(
                    x=volumes,
                    y=y[:, i],
                    mode="lines",
                    name=f"{temperatures[i]:.{temperature_decimals}f} K",
                )
            )
        plot_format(
            s_v_fig,
            "Volume (\u212B<sup>3</sup>)",
            y_label,
        )
        s_v_fig.show()


# TODO: provide choice for other EOS fitting functions
def process_debye_gruneisen(
    config_path: str,
    scaling_factor: float = 0.617,
    gruneisen_x: float = 1,
    mass_average: str = "geometric",
    volumes: np.array = None,
    temperatures: np.array = np.linspace(0, 1000, 101),
    outcar_name: str = "OUTCAR.3static",
    oszicar_name: str = "OSZICAR.3static",
    contcar_name: str = "CONTCAR.3static",
    collect_mag_data: bool = False,
    magmom_tolerance: float = 1e-12,
    total_magnetic_moment_tolerance: float = 1e-12,
    eos_fitting=eos_fit.BM4,
    plot=True,
    selected_temperatures_plot: np.array = None,
) -> tuple[np.array, np.array, int, np.array, np.array, np.array]:
    """Applies the Debye-Gruneisen model to a given configuration for which E-V curve calculations have been performed.

    Args:
        config_path: Path to the config folder
        scaling_factor: s, Scaling factor for the Debye temperature
        gruneisen_x: x = 2/3 for high temperature case and x = 1 for low temperature case
        mass_average: Type of averaging to use for the atomic masses. Can be 'arithmetic', 'geometric', or 'harmonic'
        volumes: Array of volumes to evaluate the Debye thermal properties at. If None, 100 volumes are linearly spaced between the minimum and maximum volumes
        temperatures: Array of temperatures to evaluate the Debye thermal properties at
        outcar_name: Name of the OUTCAR file
        oszicar_name: Name of the OSZICAR file
        contcar_name: Name of the CONTCAR file
        collect_mag_data: Weather or not to collect magnetic data
        magmom_tolerance: Magnetic moment tolerance for each atom
        total_magnetic_moment_tolerance: Total magnetic moment tolerance
        eos_fitting: Equation of state fitting function from the eos_fit module

        Returns:
            tuple[np.array, np.array, int, np.array, np.array, np.array]: temperatures, volumes, number of atoms, and 2D arrays with rows (columns) corresponding
            to volumes (temperatures) vibrational entropy, vibrational Helmholtz energy, vibrational heat capacity
    """

    # Extract the volume and energy from the ev_curve_series
    df = extract_configuration_data(
        path=config_path,
        outcar_name=outcar_name,
        oszicar_name=oszicar_name,
        contcar_name=contcar_name,
        collect_mag_data=collect_mag_data,
        magmom_tolerance=magmom_tolerance,
        total_magnetic_moment_tolerance=total_magnetic_moment_tolerance,
    )

    # Fit the equation of state
    volume = df["volume"]
    energy = df["energy"]
    _, eos_parameters, _, _, _ = eos_fitting(volume, energy)
    volume_0, energy_0, bulk_modulus, bulk_modulus_prime, bulk_modulus_2prime = (
        eos_parameters
    )

    s = scaling_factor
    gru_param = gruneisen_parameter(bulk_modulus_prime, gruneisen_x)

    if volumes is None:
        volume_min = volume.min() * 0.98
        volume_max = volume.max() * 1.02
        volumes = np.linspace(volume_min, volume_max, 1000)

    for dir in os.listdir(config_path):
        if dir.startswith("vol"):
            contcar = os.path.join(config_path, dir, contcar_name)
            outcar = os.path.join(config_path, dir, outcar_name)
            break

    atomic_mass = extract_average_mass(contcar, outcar, mass_average)

    theta = debye_temperature(volumes, eos_parameters, atomic_mass, gru_param, s)

    s_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    f_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    cv_vib_v_t = np.zeros((len(volumes), len(temperatures)))
    number_of_atoms = df["number_of_atoms"][0]

    for i, volume in enumerate(volumes):
        s_vib = vibrational_entropy(temperatures, theta[i], number_of_atoms)
        f_vib = vibrational_helmholtz_energy(temperatures, theta[i], number_of_atoms)
        cv_vib = vibrational_heat_capacity(temperatures, theta[i], number_of_atoms)
        s_vib_v_t[i, :] = s_vib
        f_vib_v_t[i, :] = f_vib
        cv_vib_v_t[i, :] = cv_vib

    f_vib_transposed = f_vib_v_t.T
    s_vib_transposed = s_vib_v_t.T
    cv_vib_transposed = cv_vib_v_t.T

    debye_properties = pd.DataFrame(
        {
            "temperatures": temperatures,
            "number_of_atoms": number_of_atoms,
            "scaling_factor": [s] * len(temperatures),
            "gruneisen_x": [gruneisen_x] * len(temperatures),
            "volume": [volumes] * len(temperatures),
            "f_vib": [col for col in f_vib_transposed],
            "s_vib": [col for col in s_vib_transposed],
            "cv_vib": [col for col in cv_vib_transposed],
        }
    )

    if plot:
        plot_debye(
            debye_properties, selected_temperatures_plot=selected_temperatures_plot
        )

    return debye_properties
