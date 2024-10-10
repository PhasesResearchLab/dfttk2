"""
Quasi-harmonic approximation module
"""

# Related third party imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Local application/library specific imports
import dfttk.eos_fit as eos_fit
from dfttk.plotly_format import plot_format

EV_PER_CUBIC_ANGSTROM_TO_GPA = 160.21766208  # 1 eV/Å^3  = 160.21766208 GPa


# At the moment, the function can handle either phonon or debye. Should it handle both at the same time?
def process_quasi_harmonic(
    volume_range: np.ndarray,
    eos_parameters_df: pd.DataFrame,
    harmonic_properties_fit: pd.DataFrame = None,
    debye_properties: pd.DataFrame = None,
    thermal_electronic_properties_fit: pd.DataFrame = None,
    P: int = 0,
    eos: str = "BM4",
    plot: bool = True,
    plot_type: str = "default",
    selected_temperatures_plot: list = None,
) -> pd.DataFrame:
    """Calculates the quasi-harmonic properties

    Args:
        volume_range (np.ndarray): Volume range for the quasi-harmonic calculations
        eos_parameters_df (pd.DataFrame): pandas dataframe containing the EOS parameters from the eos_fit.fit_to_all function
        harmonic_properties_fit (pd.DataFrame, optional): pandas dataframe containing the fitted harmonic properties from the fit_harmonic function
        debye_properties (pd.DataFrame, optional): pandas dataframe containing the Debye properties. Defaults to None.
        thermal_electronic_properties_fit (pd.DataFrame, optional): pandas dataframe containing the fitted thermal electronic properties. Defaults to None.
        P (int, optional): Pressure in GPa. Defaults to 0.
        plot (bool, optional): Defaults to True.
        plot_type (str, optional): Type of plots to include. Defaults to 'default'.
        selected_temperatures_plot (list, optional): List of selected temperatures to plot. Defaults to None.

    Returns:
        pd.DataFrame: pandas dataframe containing the quasi-harmonic properties
    """

    # Check that all properties have the same number of atoms
    num_atoms_eos = eos_parameters_df["number_of_atoms"].values[0]

    # Phonons only
    if harmonic_properties_fit is not None and debye_properties is None:
        num_atoms_vib = harmonic_properties_fit["number_of_atoms"].values[0]
        if num_atoms_eos != num_atoms_vib:
            raise ValueError("The number of atoms do not match")
        # Thermal electronic contribution
        if thermal_electronic_properties_fit is not None:
            num_atoms_tec = thermal_electronic_properties_fit["number_of_atoms"].values[
                0
            ]
            if num_atoms_eos != num_atoms_tec:
                raise ValueError("The number of atoms do not match")

    # Debye model only
    elif debye_properties is not None and harmonic_properties_fit is None:
        num_atoms_debye = debye_properties["number_of_atoms"].values[0]
        if num_atoms_eos != num_atoms_debye:
            raise ValueError("The number of atoms do not match")
        # Thermal electronic contribution
        if thermal_electronic_properties_fit is not None:
            num_atoms_tec = thermal_electronic_properties_fit["number_of_atoms"].values[
                0
            ]
            if num_atoms_eos != num_atoms_tec:
                raise ValueError("The number of atoms do not match")

    # EOS parameters at 0 K
    if eos == "murnaghan" or eos == "vinet" or eos == "morse":
        raise NotImplementedError(
            "Not implemented for Murnaghan, Vinet, or Morse EOS yet"
        )

    eos_parameters_one = eos_parameters_df[eos_parameters_df["eos"] == eos]
    a = eos_parameters_one["a"].values[0]
    b = eos_parameters_one["b"].values[0]
    c = eos_parameters_one["c"].values[0]
    d = eos_parameters_one["d"].values[0]
    e = eos_parameters_one["e"].values[0]

    # Get the EOS energy at 0 K corresponding to the volume range
    equation_functions = {
        "mBM4": eos_fit.mBM4_equation,
        "mBM5": eos_fit.mBM5_equation,
        "BM4": eos_fit.BM4_equation,
        "BM5": eos_fit.BM5_equation,
        "LOG4": eos_fit.LOG4_equation,
        "LOG5": eos_fit.LOG5_equation,
    }
    if eos == "mBM4" or eos == "BM4" or eos == "LOG4":
        energy_eos = equation_functions[eos](volume_range, a, b, c, d)
    elif eos == "mBM5" or eos == "BM5" or eos == "LOG5":
        energy_eos = equation_functions[eos](volume_range, a, b, c, d, e)

    # For each temperature, add energy_eos to f_vib_fit, then fit to an EOS
    f_plus_pv_list = []
    volume_range_list = []
    eos_constants_list = []
    V0_list = []
    F0_list = []
    B_list = []
    BP_list = []
    S0_list = []

    P = P / EV_PER_CUBIC_ANGSTROM_TO_GPA  # Convert GPa to eV/Å³

    if harmonic_properties_fit is not None and debye_properties is None:
        temperature_list = harmonic_properties_fit.index.tolist()

    elif debye_properties is not None and harmonic_properties_fit is None:
        temperature_list = debye_properties["temperatures"].tolist()

    eos_fit_functions = {
        "mBM4": eos_fit.mBM4,
        "mBM5": eos_fit.mBM5,
        "BM4": eos_fit.BM4,
        "BM5": eos_fit.BM5,
        "LOG4": eos_fit.LOG4,
        "LOG5": eos_fit.LOG5,
    }

    for temperature in temperature_list:
        if harmonic_properties_fit is not None and debye_properties is None:
            f_vib_poly = harmonic_properties_fit.loc[temperature]["f_vib_poly"]
            f_vib_fit = f_vib_poly(volume_range)
            f_plus_pv = energy_eos + f_vib_fit + P * volume_range

            if thermal_electronic_properties_fit is not None:
                f_el_poly = thermal_electronic_properties_fit.loc[temperature][
                    "f_el_poly"
                ]
                f_el_fit = f_el_poly(volume_range)
                f_plus_pv += f_el_fit

            f_plus_pv_list.append(f_plus_pv)
            volume_range_list.append(volume_range)

        elif debye_properties is not None and harmonic_properties_fit is None:
            # Check if the volume range is the same as the one used for the Debye model
            volume_range_debye = debye_properties[
                debye_properties["temperatures"] == temperature
            ]["volume"].values[0]
            if not np.array_equal(volume_range, volume_range_debye):
                raise ValueError(
                    "The volume range used for the Debye model is different from the one used for the EOS"
                )
            f_vib = debye_properties[debye_properties["temperatures"] == temperature][
                "f_vib"
            ].values[0]
            f_plus_pv = energy_eos + f_vib + P * volume_range

            if thermal_electronic_properties_fit is not None:
                f_el_poly = thermal_electronic_properties_fit.loc[temperature][
                    "f_el_poly"
                ]
                f_el_fit = f_el_poly(volume_range)
                f_plus_pv += f_el_fit

            f_plus_pv_list.append(f_plus_pv)
            volume_range_list.append(volume_range)

        try:
            eos_constants, eos_parameters, _, _, _ = eos_fit_functions[eos](
                volume_range, f_plus_pv
            )

        except RuntimeError as e:
            print(f"Error fitting EOS at {temperature} K: {e}")
            print(
                f"Suggestion: Try using a different EOS. Available options are: {list(eos_fit_functions.keys())}"
            )

        eos_constants_list.append(eos_constants)

        V0 = eos_parameters[0]
        F0 = eos_parameters[1]
        B = eos_parameters[2]
        BP = eos_parameters[3]

        V0_list.append(V0)
        F0_list.append(F0)
        B_list.append(B)
        BP_list.append(BP)

        if harmonic_properties_fit is not None and debye_properties is None:
            s_vib_poly = harmonic_properties_fit.loc[temperature]["s_vib_poly"]
            order = s_vib_poly.order
            s_vib_fit = s_vib_poly(volume_range)
            s_vib = s_vib_fit

        elif debye_properties is not None and harmonic_properties_fit is None:
            s_vib = debye_properties[debye_properties["temperatures"] == temperature][
                "s_vib"
            ].values[0]
            order = 2

        if thermal_electronic_properties_fit is not None:
            s_el_poly = thermal_electronic_properties_fit.loc[temperature]["s_el_poly"]
            s_el_fit = s_el_poly(volume_range)
            s_el = s_el_fit

        elif thermal_electronic_properties_fit is None:
            s_el = 0

        s = s_vib + s_el
        s_coefficients = np.polyfit(volume_range, s, order)
        s_poly = np.poly1d(s_coefficients)

        S0 = s_poly(V0)
        S0_list.append(S0)

    # Create a quasi-harmonic dataframe
    quasi_harmonic_properties = pd.DataFrame(
        data={
            "pressure": [P] * len(temperature_list),
            "number_of_atoms": [num_atoms_eos] * len(temperature_list),
            "temperature": temperature_list,
            "volume_range": volume_range_list,
            "f_plus_pv": f_plus_pv_list,
            "eos_constants": eos_constants_list,
            "V0": V0_list,
            "G0": F0_list,
            "B": B_list,
            "BP": BP_list,
            "S0": S0_list,
        }
    )

    # Calculate other properties using the finite difference method
    V0 = quasi_harmonic_properties["V0"].values
    S0 = quasi_harmonic_properties["S0"].values
    T = quasi_harmonic_properties["temperature"].values
    dV = V0[1:] - V0[:-1]
    dS = S0[1:] - S0[:-1]
    dT = T[1:] - T[:-1]

    CTE = (1 / V0[:-1]) * dV / dT * 1e6
    Cp = T[:-1] * dS / dT

    # Add a NaN value to the end of the CTE list
    CTE = np.append(CTE, np.nan)
    Cp = np.append(Cp, np.nan)

    quasi_harmonic_properties["H0"] = (
        quasi_harmonic_properties["G0"]
        + quasi_harmonic_properties["temperature"] * quasi_harmonic_properties["S0"]
    )
    quasi_harmonic_properties["CTE"] = CTE
    quasi_harmonic_properties["Cp"] = Cp

    if plot == True:
        plot_quasi_harmonic(
            quasi_harmonic_properties,
            plot_type,
            selected_temperatures_plot=selected_temperatures_plot,
        )

    return quasi_harmonic_properties


def plot_quasi_harmonic(
    quasi_harmonic_properties: pd.DataFrame,
    plot_type: str = "default",
    selected_temperatures_plot: list = None,
):
    """Plots the quasi-harmonic properties

    Args:
        quasi_harmonic_properties (pd.DataFrame): pandas dataframe containing the quasi-harmonic properties from the quasi_harmonic function
        plot_type (str, optional): Type of plots to include. Defaults to 'default'.
        selected_temperatures_plot (list, optional): List of selected temperatures to plot. Defaults to None.
    """

    temperature_list = quasi_harmonic_properties["temperature"].values
    if selected_temperatures_plot is None:
        spaces = len(temperature_list) - 1
        step = int(spaces / 10)

        selected_temperatures = temperature_list[::step]
        if selected_temperatures[-1] != temperature_list[-1]:
            selected_temperatures = np.append(
                selected_temperatures, temperature_list[-1]
            )

    else:
        selected_temperatures = selected_temperatures_plot

    scale_atoms = quasi_harmonic_properties["number_of_atoms"].iloc[0]

    if plot_type == "default" or plot_type == "all":
        # Free energy plot
        fig = go.Figure()
        for temperature in selected_temperatures:
            x = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["volume_range"].values[0]
            y = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["f_plus_pv"].values[0]

            G0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["G0"].values[0]
            V0 = quasi_harmonic_properties[
                quasi_harmonic_properties["temperature"] == temperature
            ]["V0"].values[0]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    marker=dict(size=10),
                    name=(
                        f"{int(temperature)} K"
                        if temperature % 1 == 0
                        else f"{temperature} K"
                    ),
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=[V0],
                    y=[G0],
                    mode="markers",
                    marker=dict(size=10, symbol="cross", color="black"),
                    showlegend=False,
                ),
            )
        plot_format(
            fig,
            f"Volume (Å³/{scale_atoms} atoms)",
            f"F + PV (eV/{scale_atoms} atoms)",
            width=600,
            height=600,
        )
        fig.show()

        x = temperature_list
        y_list = ["V0", "CTE", "S0", "Cp"]
        y_labels = [
            f"Volume (Å³/{scale_atoms} atoms)",
            "CTE (10<sup>-6</sup> K<sup>-1</sup>)",
            f"Entropy (eV/K/{scale_atoms} atoms)",
            f"C<sub>p</sub> (eV/K/{scale_atoms} atoms)",
        ]

        if plot_type == "all":
            y_list += ["H0", "B", "G0"]
            y_labels += [
                f"Enthalpy (eV/{scale_atoms} atoms)",
                "Bulk modulus (GPa)",
                f"Gibbs energy (eV/{scale_atoms} atoms)",
            ]

        for y_value, y_label in zip(y_list, y_labels):
            fig = go.Figure()
            x = temperature_list
            y = quasi_harmonic_properties[y_value].values

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    marker=dict(size=10),
                ),
            )
            plot_format(
                fig,
                f"Temperature (K)",
                y_label,
                width=600,
                height=600,
            )
            fig.show()
