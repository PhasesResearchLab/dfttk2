"""
Calculates the thermal electronic contribution to the Helmholtz energy.
"""

# Standard Library Imports
import os

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.constants
from scipy.special import expit
from natsort import natsorted
from scipy.interpolate import UnivariateSpline

# Local application/library specific imports
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.electronic_structure.core import Spin

# DFTTK imports
from dfttk.data_extraction import extract_volume
from dfttk.plotly_format import plot_format

BOLTZMANN_CONSTANT = (
    scipy.constants.Boltzmann / scipy.constants.electron_volt
)  # The Boltzmann constant in eV/K


def read_total_electron_dos(path: str, plot: bool = False) -> pd.DataFrame:
    """Reads the total electron DOS from vasprun.xml files

    Args:
        path (str): path to the directory containing the elec_folders
        plot (bool, optional): plots the total electron DOS for different volumes. Defaults to False.

    Returns:
        pd.DataFrame: dataframe containing the electron DOS data
    """

    elec_folders = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder)) and folder.startswith("elec")
    ]
    elec_folders = natsorted(elec_folders)

    volume_list = []
    num_atoms_list = []
    energy_minus_fermi_energy_list = []
    total_dos_list = []
    for elec_folder in elec_folders:
        volume = extract_volume(os.path.join(path, elec_folder, "CONTCAR.elec_dos"))

        vasprun_path = os.path.join(path, elec_folder, "vasprun.xml.elec_dos")
        vasprun = Vasprun(vasprun_path)

        volume = extract_volume(os.path.join(path, elec_folder, "CONTCAR.elec_dos"))
        num_atoms = vasprun.final_structure.num_sites

        # TODO: Implement this for magnetic systems as well
        complete_dos = vasprun.complete_dos
        energy = complete_dos.energies
        total_dos = complete_dos.densities[Spin.up]

        fermi_energy = vasprun.efermi
        energy_minus_fermi_energy = energy - fermi_energy

        volume_list.append(volume)
        num_atoms_list.append(num_atoms)
        energy_minus_fermi_energy_list.append(energy_minus_fermi_energy)
        total_dos_list.append(total_dos)

    electron_dos_data = pd.DataFrame(
        {
            "volume": volume_list,
            "number_of_atoms": num_atoms_list,
            "energy_minus_fermi_energy": energy_minus_fermi_energy_list,
            "total_dos": total_dos_list,
        }
    )
    electron_dos_data = electron_dos_data.sort_values(by="volume")
    electron_dos_data = electron_dos_data.reset_index(drop=True)

    if plot:
        plot_total_electron_dos(electron_dos_data)

    return electron_dos_data


def plot_total_electron_dos(electron_dos_data: pd.DataFrame):
    """Plots the total electron DOS for different volumes

    Args:
        pd.DataFrame: dataframe containing the electron DOS data
    """

    fig = go.Figure()
    for i in range(len(electron_dos_data)):
        fig.add_trace(
            go.Scatter(
                x=electron_dos_data["energy_minus_fermi_energy"].iloc[i],
                y=electron_dos_data["total_dos"].iloc[i],
                mode="lines",
                name=f"{electron_dos_data['volume'].iloc[i]} Å<sup>3</sup>",
                showlegend=True,
            )
        )
    plot_format(
        fig,
        xtitle="E - E<sub>F</sub> (eV)",
        ytitle=f"DOS (states/eV/{electron_dos_data['number_of_atoms'].iloc[i]} atoms)",
    )
    fig.show()


def fit_electron_dos(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    energy_range: list[float, float],
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fits the electron DOS with a spline

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        energy_range (list[float, float]): energy range to fit the electron DOS
        resolution (float): energy resolution for the spline

    Returns:
        tuple[np.ndarray, np.ndarray]: fitted energy and DOS values
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    # Filter the energy and dos values within the energy range
    filtered_indices = (energy >= energy_range[0]) & (energy <= energy_range[1])
    filtered_energy = energy[filtered_indices]
    filtered_dos = dos[filtered_indices]

    # Fit the filtered energy and dos values with a spline
    spline = UnivariateSpline(filtered_energy, filtered_dos, s=0)
    energy_fit = np.arange(energy_range[0], energy_range[1] + resolution, resolution)
    dos_fit = spline(energy_fit)

    return energy_fit, dos_fit


def fermi_dirac_distribution(
    energy: pd.Series | np.ndarray,
    chemical_potential: float,
    temperature: float,
    plot=False,
) -> np.ndarray:
    """Calculates the Fermi-Dirac distribution function

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        chemical_potential (float): chemical potential for a given volume and temperature
        temperature (float): temperature range
        plot (bool, optional): plots the Fermi-Dirac distribution function vs. energy for a
        given temperature and chemical potential. Defaults to False.

    Raises:
        ValueError: Temperature cannot be less than 0 K

    Returns:
        np.ndarray: Fermi-Dirac distribution function values
    """

    # Check if energy is a pandas Series and convert to NumPy array if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    if temperature == 0:
        fermi_dist = np.where(energy < chemical_potential, 1, 0)

    if temperature > 0:
        # Note that expit(x) = 1/(1+exp(-x))
        fermi_dist = expit(
            -(energy - chemical_potential) / (BOLTZMANN_CONSTANT * temperature)
        )

    if plot:
        plot_fermi_dirac_distribution(
            energy, chemical_potential, temperature, fermi_dist
        )

    return fermi_dist


def plot_fermi_dirac_distribution(
    energy: pd.Series | np.ndarray,
    chemical_potential: float,
    temperature: float,
    fermi_dist: np.ndarray,
):
    """Plots the Fermi-Dirac distribution function vs. energy for a given temperature and
    chemical potential

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        chemical_potential (float): chemical potential for a given volume and temperature
        temperature (float): temperature
        fermi_dist (np.ndarray):  Fermi-Dirac distribution function values
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=fermi_dist, mode="lines"))
    fig.update_layout(
        title=dict(
            text=f"T = {temperature} K, &mu; = {chemical_potential} eV",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="f (E, T, V) ")
    fig.show()


def calculate_num_electrons(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    chemical_potential: float,
    temperature: float,
) -> float:
    """_summary_

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        chemical_potential (float): chemical potential for a given volume and temperature
        temperature (float): temperature

    Raises:
        ValueError: Temperature cannot be less than 0 K

    Returns:
        float: number of electrons
    """

    if temperature < 0:
        raise ValueError("Temperature cannot be less than 0 K")

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    fermi_dist = fermi_dirac_distribution(energy, chemical_potential, temperature)
    integrand = dos * fermi_dist
    num_electrons = np.trapz(integrand, energy)

    return num_electrons


def calculate_chemical_potential(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    temperature: float,
    min_chemical_potential: float = -0.2,
    max_chemical_potential: float = 0.2,
) -> float:
    """Calculates the chemical potential at a given temperature and volume

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        temperature (float): temperature
        min_chemical_potential (float, optional): starting search for the chemical potential. Defaults to -0.2.
        max_chemical_potential (float, optional): end search for the chemical potential. Defaults to 0.2.

    Returns:
        float: chemical potential at a given temperature and volume
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    num_electrons_0K = calculate_num_electrons(energy, dos, 0, 0)

    # Find the chemical potential at temperature such that the number of electrons matches that at 0 K
    chemical_potential = min_chemical_potential
    num_electrons = calculate_num_electrons(
        energy, dos, chemical_potential, temperature
    )
    while (
        abs(num_electrons - num_electrons_0K) > 1e-2
        and chemical_potential < max_chemical_potential
    ):
        chemical_potential += 0.01
        num_electrons = calculate_num_electrons(
            energy, dos, chemical_potential, temperature
        )

    return chemical_potential


def calculate_internal_energy(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    temperature_range: np.ndarray,
    resolution: float = 0.001,
    plot: bool = False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the internal energy

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        temperature_range (np.ndarray): temperature range
        resolution (float, optional): energy resolution for the spline. Defaults to 0.001.
        plot (bool, optional): plots the integrand vs energy of the internal energy equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the internal energy equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the internal energy
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    energy_fit, dos_fit = fit_electron_dos(
        energy, dos, [np.min(energy), np.max(energy)], resolution
    )
    integrand_1_list = []
    filtered_energy_list = []
    integrand_2_list = []
    E_el_list = []
    for temperature in temperature_range:
        chemical_potential = calculate_chemical_potential(
            energy_fit, dos_fit, temperature
        )
        fermi_dist = fermi_dirac_distribution(
            energy_fit, chemical_potential, temperature
        )

        integrand_1 = dos_fit * fermi_dist * energy_fit
        integrand_1_list.append(integrand_1)
        integral_1 = np.trapz(integrand_1, energy_fit)

        # Only integrate over energy levels less than the chemical potential
        mask = energy_fit < chemical_potential
        filtered_energy = energy_fit[mask]
        filtered_energy_list.append(filtered_energy)
        filtered_dos = dos_fit[mask]

        integrand_2 = filtered_dos * filtered_energy
        integrand_2_list.append(integrand_2)
        integral_2 = np.trapz(integrand_2, filtered_energy)

        E_el = integral_1 - integral_2
        E_el_list.append(E_el)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]

            plot_internal_energy_integral(
                energy_fit,
                integrand_1_list[index],
                filtered_energy_list[index],
                integrand_2_list[index],
                plot_temperature,
            )

    return E_el_list


def plot_internal_energy_integral(
    energy: np.ndarray,
    integrand_1: np.ndarray,
    filtered_energy: np.ndarray,
    integrand_2: np.ndarray,
    plot_temperature: float,
):
    """Plots the integrand vs energy of the internal energy equation

    Args:
        energy (np.ndarray): energy values from the electron DOS
        integrand_1 (np.ndarray): integrand 1 from the internal energy equation
        filtered_energy (np.ndarray): filtered energy values from the electron DOS
        integrand_2 (np.ndarray): integrand 2 from the internal energy equation
        plot_temperature (float): temperature
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=integrand_1, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 1"
    )
    fig.show()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_energy, y=integrand_2, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="E<sub>el</sub> integrand 2"
    )
    fig.show()


def calculate_entropy(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    temperature_range: np.ndarray,
    energy_range: list[float, float] = [-2, 2],
    resolution: float = 0.0001,
    plot: bool = False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the entropy

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        temperature_range (np.ndarray): temperature range
        energy_range (list[float, float], optional): energy range to fit the electron DOS. Defaults to [-2, 2].
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the entropy equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the entropy equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the entropy
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    energy_fit, dos_fit = fit_electron_dos(energy, dos, energy_range, resolution)

    integrand_list = []
    S_el_list = []
    for temperature in temperature_range:
        if temperature == 0:
            S_el = 0
            S_el_list.append(S_el)

            integrand = np.zeros_like(energy_fit)
            integrand_list.append(integrand)

        elif temperature > 0:
            chemical_potential = calculate_chemical_potential(
                energy_fit, dos_fit, temperature
            )
            fermi_dist = fermi_dirac_distribution(
                energy_fit, chemical_potential, temperature
            )

            # The limit of f ln f + (1-f) ln (1-f) as f approaches 0 or 1 is 0
            mask = (fermi_dist == 0) | (fermi_dist == 1)
            integrand = np.zeros_like(fermi_dist)
            integrand[~mask] = dos_fit[~mask] * (
                fermi_dist[~mask] * np.log(fermi_dist[~mask])
                + (1 - fermi_dist[~mask]) * np.log(1 - fermi_dist[~mask])
            )
            integrand[mask] = 0
            integrand_list.append(integrand)

            S_el = -BOLTZMANN_CONSTANT * np.trapz(integrand, energy_fit)
            S_el_list.append(S_el)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]
            plot_entropy_integral(energy_fit, integrand_list[index], plot_temperature)

    return S_el_list


def plot_entropy_integral(
    energy: np.ndarray, integrand: np.ndarray, plot_temperature: float
):
    """Plots the integrand vs energy of the entropy equation

    Args:
        energy (np.ndarray): energy values from the electron DOS
        integrand (np.ndarray): integrand from the entropy equation
        plot_temperature (float): temperature
    """

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=energy, y=-BOLTZMANN_CONSTANT * integrand, mode="markers")
    )
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="S<sub>el</sub> integrand")
    fig.show()


# TODO: Equation is from the MATLAB code. Double check this.
def calculate_heat_capacity(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    temperature_range: np.ndarray,
    energy_range: list[float, float] = [-2, 2],
    resolution: float = 0.0001,
    plot=False,
    plot_temperatures: np.ndarray = None,
) -> list:
    """Calculates the thermal electronic contribution to the heat capacity

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        temperature_range (np.ndarray): temperature range
        energy_range (list[float, float], optional): energy range to fit the electron DOS. Defaults to [-2, 2].
        resolution (float, optional): energy resolution for the spline. Defaults to 0.0001.
        plot (bool, optional): plots the integrand vs energy of the heat capacity equation. Defaults to False.
        plot_temperatures (np.ndarray, optional): temperatures to plot the integrand vs energy of the heat capacity equation. Defaults to None.

    Returns:
        list: thermal electronic contribution to the heat capacity
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    energy_fit, dos_fit = fit_electron_dos(energy, dos, energy_range, resolution)

    integrand_list = []
    Cv_el_list = []

    for temperature in temperature_range:
        if temperature == 0:
            Cv_el = 0
            Cv_el_list.append(Cv_el)

            integrand = np.zeros_like(energy_fit)
            integrand_list.append(integrand)

        elif temperature > 0:
            chemical_potential = calculate_chemical_potential(
                energy_fit, dos_fit, temperature
            )
            fermi_dist = fermi_dirac_distribution(
                energy_fit, chemical_potential, temperature
            )

            # The limit of (1 / f - 1) * (f * (E - mu) / T) ** 2 as f approaches 0 or 1 is 0
            mask = (fermi_dist == 0) | (fermi_dist == 1)
            integrand = np.zeros_like(fermi_dist)
            integrand[~mask] = dos_fit[~mask] * (
                (1 / fermi_dist[~mask] - 1)
                * (
                    fermi_dist[~mask]
                    * (energy_fit[~mask] - chemical_potential)
                    / temperature
                )
                ** 2
                / BOLTZMANN_CONSTANT
            )
            integrand[mask] = 0
            integrand_list.append(integrand)

            Cv_el = np.trapz(integrand, energy_fit)
            Cv_el_list.append(Cv_el)

    if plot:
        for plot_temperature in plot_temperatures:
            index = np.where(temperature_range == plot_temperature)[0][0]
            plot_heat_capacity_integral(
                energy_fit, integrand_list[index], plot_temperature
            )

    return Cv_el_list


def plot_heat_capacity_integral(
    energy: np.ndarray, integrand: np.ndarray, plot_temperature: float
):
    """plots the integrand vs energy of the heat capacity equation

    Args:
        energy (np.ndarray): energy values from the electron DOS
        integrand (np.ndarray): integrand from the heat capacity equation
        plot_temperature (float): temperature
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energy, y=integrand, mode="markers"))
    fig.update_layout(
        title=dict(
            text=f"T = {plot_temperature} K",
            font=dict(size=20, color="rgb(0,0,0)"),
        ),
        margin=dict(t=130),
    )
    plot_format(
        fig, xtitle="E - E<sub>F</sub> (eV)", ytitle="C<sub>v,el</sub> integrand"
    )
    fig.show()


def calculate_free_energy(
    energy: pd.Series | np.ndarray,
    dos: pd.Series | np.ndarray,
    temperature_range: np.ndarray,
) -> list:
    """Calculates the thermal electronic contribution to the Helmholtz energy

    Args:
        energy (pd.Series | np.ndarray): energy values from the electron DOS
        dos (pd.Series | np.ndarray): electron DOS values
        temperature_range (np.ndarray): temperature range

    Returns:
        list: thermal electronic contribution to the Helmholtz energy
    """

    # Check if energy and dos are pandas Series and convert to NumPy arrays if necessary
    if isinstance(energy, pd.Series):
        energy = energy.values[0]
    if isinstance(dos, pd.Series):
        dos = dos.values[0]

    E_el_list = calculate_internal_energy(energy, dos, temperature_range)
    S_el_list = calculate_entropy(energy, dos, temperature_range)
    F_el = E_el_list - temperature_range * S_el_list

    return F_el.tolist()


def thermal_electronic(
    electron_dos_data: pd.DataFrame,
    temperature_range: np.ndarray,
    order: int = 2,
    plot: bool = True,
    selected_temperatures_plot: np.ndarray = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates the thermal electronic properties

    Args:
        electron_dos_data (pd.DataFrame): dataframe containing the electron DOS data
        temperature_range (np.ndarray): temperature range
        order (int, optional): order of polynomial to fit thermal electronic properties vs. volume for a fixed temperature. Defaults to 2.
        plot (bool, optional): plots the thermal electronic properties vs. temperature and vs. volume. Defaults to True.
        selected_temperatures_plot (np.ndarray, optional): selected temperatures to plot. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: dataframes containing the thermal electronic properties and the fitted thermal electronic properties
    """

    E_el_list = []
    S_el_list = []
    Cv_el_list = []
    F_el_list = []

    volumes = electron_dos_data["volume"].unique()
    for volume in volumes:
        row = electron_dos_data.loc[electron_dos_data["volume"] == volume]
        energy = row["energy_minus_fermi_energy"]
        dos = row["total_dos"]

        E_el = calculate_internal_energy(energy, dos, temperature_range)
        S_el = calculate_entropy(energy, dos, temperature_range)
        Cv_el = calculate_heat_capacity(energy, dos, temperature_range)
        F_el = calculate_free_energy(energy, dos, temperature_range)

        E_el_list.append(E_el)
        S_el_list.append(S_el)
        Cv_el_list.append(Cv_el)
        F_el_list.append(F_el)

    # Flatten the lists of lists using list comprehension
    E_el_list = [item for sublist in E_el_list for item in sublist]
    S_el_list = [item for sublist in S_el_list for item in sublist]
    Cv_el_list = [item for sublist in Cv_el_list for item in sublist]
    F_el_list = [item for sublist in F_el_list for item in sublist]

    number_of_atoms = electron_dos_data["number_of_atoms"].unique()[0]
    thermal_electronic_properties = pd.DataFrame(
        {
            "number_of_atoms": np.repeat(
                number_of_atoms, len(temperature_range) * len(volumes)
            ),
            "volume": np.repeat(volumes, len(temperature_range)),
            "temperature": np.tile(temperature_range, len(volumes)),
            "f_el": F_el_list,
            "e_el": E_el_list,
            "s_el": S_el_list,
            "cv_el": Cv_el_list,
        }
    )

    thermal_electronic_properties_fit = fit_thermal_electronic(
        thermal_electronic_properties, order
    )

    if plot == True:
        plot_thermal_electronic(thermal_electronic_properties)
        plot_thermal_electronic_properties_fit(
            thermal_electronic_properties_fit, selected_temperatures_plot
        )

    return thermal_electronic_properties, thermal_electronic_properties_fit


def plot_thermal_electronic(thermal_electronic_properties: pd.DataFrame):
    """Plots the thermal electronic properties vs. temperature and vs. volume

    Args:
        thermal_electronic_properties (pd.DataFrame): dataframe containing the thermal electronic properties
    """

    volumes = thermal_electronic_properties["volume"].unique()
    number_of_atoms = thermal_electronic_properties["number_of_atoms"].unique()[0]

    y_values = ["f_el", "s_el", "cv_el"]
    for y_value in y_values:
        fig = go.Figure()
        for volume in volumes:
            temperature = thermal_electronic_properties[
                thermal_electronic_properties["volume"] == volume
            ]["temperature"]
            y_data = thermal_electronic_properties[
                thermal_electronic_properties["volume"] == volume
            ][y_value]

            fig.add_trace(
                go.Scatter(
                    x=temperature,
                    y=y_data,
                    mode="lines",
                    name=f"{volume} Å³",
                    showlegend=True,
                )
            )

        if y_value == "f_el":
            y_title = f"F<sub>el</sub> (eV/{number_of_atoms} atoms)"
        elif y_value == "s_el":
            y_title = f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)"
        elif y_value == "cv_el":

            y_title = f"C<sub>v, el</sub> (eV/K/{number_of_atoms} atoms)"

        plot_format(fig, "Temperature (K)", y_title)
        fig.show()


def fit_thermal_electronic(
    thermal_electronic_properties: pd.DataFrame, order: int
) -> pd.DataFrame:
    """Fits the thermal electronic properties vs. volume for various fixed temperatures to a polynomial

    Args:
        thermal_electronic_properties (pd.DataFrame): dataframe containing the thermal electronic properties
        order (int): order of polynomial

    Returns:
        pd.DataFrame: dataframe containing the fitted thermal electronic properties
    """

    volume_fit_list = []
    F_el_fit_list = []
    S_el_fit_list = []
    Cv_el_fit_list = []
    F_el_polynomial_list = []
    S_el_polynomial_list = []
    Cv_el_polynomial_list = []

    thermal_electronic_properties_fit = thermal_electronic_properties.groupby(
        "temperature"
    ).agg(list)
    temperatures = thermal_electronic_properties_fit.index.tolist()

    for temperature in temperatures:
        volume = thermal_electronic_properties_fit.loc[temperature]["volume"]

        F_el = thermal_electronic_properties_fit.loc[temperature]["f_el"]
        S_el = thermal_electronic_properties_fit.loc[temperature]["s_el"]
        Cv_el = thermal_electronic_properties_fit.loc[temperature]["cv_el"]

        F_el_coefficients = np.polyfit(volume, F_el, order)
        S_el_coefficients = np.polyfit(volume, S_el, order)
        Cv_el_coefficients = np.polyfit(volume, Cv_el, order)

        F_el_polynomial = np.poly1d(F_el_coefficients)
        S_el_polynomial = np.poly1d(S_el_coefficients)
        Cv_el_polynomial = np.poly1d(Cv_el_coefficients)

        F_el_polynomial_list.append(F_el_polynomial)
        S_el_polynomial_list.append(S_el_polynomial)
        Cv_el_polynomial_list.append(Cv_el_polynomial)

        volume_fit = np.linspace(min(volume) * 0.98, max(volume) * 1.02, 1000)
        F_el_fit = F_el_polynomial(volume_fit)
        S_el_fit = S_el_polynomial(volume_fit)
        Cv_el_fit = Cv_el_polynomial(volume_fit)

        volume_fit_list.append(volume_fit)
        F_el_fit_list.append(F_el_fit)
        S_el_fit_list.append(S_el_fit)
        Cv_el_fit_list.append(Cv_el_fit)

    thermal_electronic_properties_fit["number_of_atoms"] = (
        thermal_electronic_properties_fit["number_of_atoms"].values[0][0]
    )
    thermal_electronic_properties_fit["volume_fit"] = volume_fit_list

    thermal_electronic_properties_fit["f_el_fit"] = F_el_fit_list
    thermal_electronic_properties_fit["s_el_fit"] = S_el_fit_list
    thermal_electronic_properties_fit["cv_el_fit"] = Cv_el_fit_list
    thermal_electronic_properties_fit["f_el_poly"] = F_el_polynomial_list
    thermal_electronic_properties_fit["s_el_poly"] = S_el_polynomial_list
    thermal_electronic_properties_fit["cv_el_poly"] = Cv_el_polynomial_list

    return thermal_electronic_properties_fit


def plot_thermal_electronic_properties_fit(
    thermal_electronic_properties_fit: pd.DataFrame,
    selected_temperatures_plot: np.ndarray = None,
):
    """Plots the fitted thermal electronic properties vs. volume for various fixed temperatures

    Args:
        thermal_electronic_properties_fit (pd.DataFrame): dataframe containing the fitted thermal electronic properties
        selected_temperatures_plot (np.ndarray, optional): selected temperatures to plot. Defaults to None.
    """

    number_of_atoms = thermal_electronic_properties_fit["number_of_atoms"].iloc[0]
    temperature_list = thermal_electronic_properties_fit.index.values
    if selected_temperatures_plot is None:
        indices = np.linspace(0, len(temperature_list) - 1, 5, dtype=int)
        selected_temperatures_plot = np.array([temperature_list[j] for j in indices])

    y_values = [
        ("f_el", "f_el_fit"),
        ("s_el", "s_el_fit"),
        ("cv_el", "cv_el_fit"),
    ]
    for y_value, y_value_fit in y_values:
        fig = go.Figure()
        colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        colors = [
            f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {1})"
            for color in colors
        ]

        i = 0
        for i, temperature in enumerate(selected_temperatures_plot):
            x = thermal_electronic_properties_fit.loc[temperature]["volume"]
            y = thermal_electronic_properties_fit.loc[temperature][y_value]
            x_fit = thermal_electronic_properties_fit.loc[temperature]["volume_fit"]
            y_fit = thermal_electronic_properties_fit.loc[temperature][y_value_fit]

            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    line=dict(color=color),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=color),
                    name=f"{temperature} K",
                    showlegend=True,
                )
            )
            i += 1

        if y_value == "f_el":
            y_title = f"F<sub>el</sub> (eV/{number_of_atoms} atoms)"
        elif y_value == "s_el":
            y_title = f"S<sub>el</sub> (eV/K/{number_of_atoms} atoms)"
        elif y_value == "cv_el":

            y_title = f"C<sub>v, el</sub> (eV/K/{number_of_atoms} atoms)"

        plot_format(fig, f"Volume (Å³/{number_of_atoms} atoms)", y_title)
        fig.show()
