"""
Store results on MongoDB.
"""

# Standard library imports
import json
import os

# Third-party library imports
import pandas as pd
from bson import ObjectId
from natsort import natsorted
from pymongo import MongoClient
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar

# DFTTK imports
from dfttk.workflows import custodian_errors_location
from dfttk.data_extraction import extract_volume


# TODO: add docstrings
class MongoDBStorage:

    # Establish connection to MongoDB
    def __init__(
        self, path: str, connection_string: str, db="DFTTK", collection="community"
    ):
        self.cluster = MongoClient(connection_string)
        self.db = self.cluster[db]
        self.collection = self.db[collection]
        self.path = path

    def store_ev_curve(
        self,
        config: str,
        df: pd.DataFrame,
        eos_parameters_df: pd.DataFrame,
        eos: str,
        multiplicity: int = 0,
        object_id: str = None,
    ):
        vol_folders = [
            f
            for f in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, f)) and f.startswith("vol")
        ]
        vol_folders = natsorted(vol_folders)

        # material
        poscar = Poscar.from_file(
            os.path.join(self.path, vol_folders[0], "POSCAR.1relax")
        )
        reduced_formula = poscar.structure.composition.reduced_formula
        number_of_atoms = poscar.structure.composition.num_atoms

        # properties
        ## EV curve
        natoms = df["number_of_atoms"].unique()[0].item()

        # WARNING: If there are errors in all vol_folders, this will not work
        vol_folders_errors, __ = custodian_errors_location(self.path)
        vol_folders_no_errors = [f for f in vol_folders if f not in vol_folders_errors]

        incar_relax = Incar.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "INCAR.1relax")
        )
        incar_static = Incar.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "INCAR.3static")
        )
        kpoints = Kpoints.from_file(
            os.path.join(self.path, vol_folders_no_errors[0], "KPOINTS.1relax")
        )

        # WARNING: the pymatgen Potcar class reads in PBE and PBE_54 as the same PBE functional
        # The following code will override that behavior
        # Issue: https://github.com/materialsproject/pymatgen/issues/3951
        # TODO: Should be able to distinguish between PBE and PBE_52 as well
        potcar_path = os.path.join(self.path, vol_folders_no_errors[0], "POTCAR")
        with open(potcar_path, "r") as file:
            lines = file.readlines()
            fourth_line = lines[3].strip()

        if "SHA256" in fourth_line:
            potcar = Potcar.from_file(potcar_path)
            potcar.functional = "PBE_54"
        else:
            potcar = Potcar.from_file(potcar_path)

        error_dict = {}
        for f in vol_folders_errors:
            file_path = os.path.join(self.path, f, "custodian.json")
            with open(file_path, "r") as file:
                error_dict[f] = json.load(file)

        ev_curve_properties = {}
        ev_curve_properties["natoms"] = natoms
        ev_curve_properties["VASP input"] = {
            "INCAR relax": incar_relax,
            "INCAR static": incar_static,
            "KPOINTS": kpoints.as_dict(),
            "POTCAR": potcar.as_dict(),
            "errors": error_dict,
        }

        volume = df["volume"].to_list()
        energy = df["energy"].to_list()
        space_group = df["space_group"].to_list()

        poscar_list = []
        for vol_folder in vol_folders:
            poscar = Poscar.from_file(
                os.path.join(self.path, vol_folder, "POSCAR.3static")
            )
            poscar_list.append(poscar.as_dict())

        eos_parameters_df = eos_parameters_df[eos_parameters_df["eos"] == eos]
        eos_dict = eos_parameters_df.to_dict(orient="records")[0]

        document = {
            "material": {
                "config": config,
                "reduced formula": reduced_formula,
                "number of atoms": number_of_atoms,
                "multiplicity": multiplicity,
            },
            "properties": {
                "EV curve": {
                    "natoms": natoms,
                    "VASP input": {
                        "INCAR relax": incar_relax,
                        "INCAR static": incar_static,
                        "KPOINTS": kpoints.as_dict(),
                        "POTCAR": potcar.as_dict(),
                        "errors": error_dict,
                    },
                    "volume": volume,
                    "energy": energy,
                    "space group": space_group,
                    "POSCAR": poscar_list,
                    "EOS fit": eos_dict,
                }
            },
        }

        if object_id:
            result = self.collection.update_one(
                {"_id": ObjectId(object_id)}, {"$set": document}
            )
            if result.matched_count > 0:
                print(f"Document with ID {object_id} updated.")
            else:
                print(f"No document found with ID {object_id}.")
        else:
            self.collection.insert_one(document)
            print("New document inserted.")

        return document

    def store_fvib_debye(self, object_id: str, debye_properties: pd.DataFrame):

        natoms_debye = debye_properties["number_of_atoms"].unique()[0].item()
        min_temperature = debye_properties["temperatures"].min().item()
        max_temperature = debye_properties["temperatures"].max().item()
        dT = debye_properties["temperatures"].diff().unique()[1].item()

        volume_range = debye_properties["volume"].to_list()[0]
        min_volume = min(volume_range)
        max_volume = max(volume_range)

        scaling_factor = debye_properties["scaling_factor"].unique()[0].item()
        gruneisen_x = debye_properties["gruneisen_x"].unique()[0].item()

        fvib_debye_data = {
            "$set": {
                "properties.Fvib debye": {
                    "natoms debye": natoms_debye,
                    "temperature": {
                        "min": min_temperature,
                        "max": max_temperature,
                        "dT": dT,
                    },
                    "volume range": {"min": min_volume, "max": max_volume},
                    "scaling factor": scaling_factor,
                    "x": gruneisen_x,
                }
            }
        }
        document_id = ObjectId(object_id)
        self.collection.update_one({"_id": document_id}, fvib_debye_data)

        return fvib_debye_data

    def store_fvib_phonons(self, object_id: str, harmonic_properties_fit: pd.DataFrame):

        # phonons
        phonon_folders = [
            f
            for f in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, f)) and f.startswith("phonon")
        ]
        phonon_folders = natsorted(phonon_folders)

        poscar = Poscar.from_file(
            os.path.join(self.path, phonon_folders[0], "POSCAR.1relax")
        )
        natoms_phonons = poscar.structure.num_sites

        __, phonon_folders_errors = custodian_errors_location(self.path)
        phonon_folders_no_errors = [
            f for f in phonon_folders if f not in phonon_folders_errors
        ]

        # WARNING: If there are errors in all phonon_folders, this will not work
        incar_relax = Incar.from_file(
            os.path.join(self.path, phonon_folders_no_errors[0], "INCAR.1relax")
        )
        incar_phonons = Incar.from_file(
            os.path.join(self.path, phonon_folders_no_errors[0], "INCAR.2phonons")
        )
        kpoints = Kpoints.from_file(
            os.path.join(self.path, phonon_folders_no_errors[0], "KPOINTS.1relax")
        )

        # WARNING: the pymatgen Potcar class reads in PBE and PBE_54 as the same PBE functional
        # The following code will override that behavior
        # Issue: https://github.com/materialsproject/pymatgen/issues/3951
        # TODO: Should be able to distinguish between PBE and PBE_52 as well
        potcar_path = os.path.join(self.path, phonon_folders_no_errors[0], "POTCAR")
        with open(potcar_path, "r") as file:
            lines = file.readlines()
            fourth_line = lines[3].strip()

        if "SHA256" in fourth_line:
            potcar = Potcar.from_file(potcar_path)
            potcar.functional = "PBE_54"
        else:
            potcar = Potcar.from_file(potcar_path)

        error_dict = {}
        for f in phonon_folders_errors:
            file_path = os.path.join(self.path, f, "custodian.json")
            with open(file_path, "r") as file:
                error_dict[f] = json.load(file)

        volume_list = []
        for phonon_folder in phonon_folders:
            volume = extract_volume(
                os.path.join(self.path, phonon_folder, "POSCAR.1relax")
            )
            volume_list.append(volume)

        poscar_list = []
        for phonon_folder in phonon_folders:
            poscar = Poscar.from_file(
                os.path.join(self.path, phonon_folder, "POSCAR.1relax")
            )
            poscar_list.append(poscar.as_dict())

        # harmonic
        natoms_harmonic = harmonic_properties_fit["number_of_atoms"].unique()[0].item()

        harmonic_properties_fit_reset = harmonic_properties_fit.reset_index()
        min_temperature = harmonic_properties_fit_reset["temperature"].min().item()
        max_temperature = harmonic_properties_fit_reset["temperature"].max().item()
        dT = harmonic_properties_fit_reset["temperature"].diff().unique()[1].item()

        volume_range = harmonic_properties_fit_reset["volume_fit"].to_list()[0]
        min_volume = min(volume_range)
        max_volume = max(volume_range)

        polynomial_columns = [
            "f_vib_poly",
            "s_vib_poly",
            "cv_vib_poly",
        ]
        polynomial_data = harmonic_properties_fit_reset[polynomial_columns]
        f_vib_coefficients = polynomial_data["f_vib_poly"].to_list()
        s_vib_coefficients = polynomial_data["s_vib_poly"].to_list()
        cv_vib_coefficients = polynomial_data["cv_vib_poly"].to_list()

        for i in range(len(f_vib_coefficients)):
            f_vib_coefficients[i] = f_vib_coefficients[i].coefficients.tolist()
            s_vib_coefficients[i] = s_vib_coefficients[i].coefficients.tolist()
            cv_vib_coefficients[i] = cv_vib_coefficients[i].coefficients.tolist()

        fvib_phonons_data = {
            "$set": {
                "properties.Fvib phonons": {
                    "natoms phonons": natoms_phonons,
                    "VASP input": {
                        "INCAR relax": incar_relax,
                        "INCAR phonons": incar_phonons,
                        "KPOINTS": kpoints.as_dict(),
                        "POTCAR": potcar.as_dict(),
                        "errors": error_dict,
                    },
                    "volume": volume_list,
                    "POSCAR": poscar_list,
                    "natoms harmonic": natoms_harmonic,
                    "temperature": {
                        "min": min_temperature,
                        "max": max_temperature,
                        "dT": dT,
                    },
                    "volume range": {"min": min_volume, "max": max_volume},
                    "Fvib coefficients": f_vib_coefficients,
                    "Svib coefficients": s_vib_coefficients,
                    "Cv vib coefficients": cv_vib_coefficients,
                }
            }
        }

        document_id = ObjectId(object_id)
        self.collection.update_one({"_id": document_id}, fvib_phonons_data)

        return fvib_phonons_data

    def store_fel(
        self, object_id: str, thermal_electronic_properties_fit: pd.DataFrame
    ):

        elec_folders = [
            f
            for f in os.listdir(self.path)
            if os.path.isdir(os.path.join(self.path, f)) and f.startswith("elec")
        ]
        elec_folders = natsorted(elec_folders)

        poscar = Poscar.from_file(
            os.path.join(self.path, elec_folders[0], "POSCAR.elec_dos")
        )
        natoms_elec = poscar.structure.num_sites

        __, elec_folders_errors = custodian_errors_location(self.path)
        elec_folders_no_errors = [
            f for f in elec_folders if f not in elec_folders_errors
        ]

        # WARNING: If there are errors in all elec_folders, this will not work
        incar_elec_dos = Incar.from_file(
            os.path.join(self.path, elec_folders_no_errors[0], "INCAR.elec_dos")
        )
        kpoints = Kpoints.from_file(
            os.path.join(self.path, elec_folders_no_errors[0], "KPOINTS.elec_dos")
        )

        # WARNING: the pymatgen Potcar class reads in PBE and PBE_54 as the same PBE functional
        # The following code will override that behavior
        # Issue: https://github.com/materialsproject/pymatgen/issues/3951
        # TODO: Should be able to distinguish between PBE and PBE_52 as well
        potcar_path = os.path.join(self.path, elec_folders_no_errors[0], "POTCAR")
        with open(potcar_path, "r") as file:
            lines = file.readlines()
            fourth_line = lines[3].strip()

        if "SHA256" in fourth_line:
            potcar = Potcar.from_file(potcar_path)
            potcar.functional = "PBE_54"
        else:
            potcar = Potcar.from_file(potcar_path)

        error_dict = {}
        for f in elec_folders_errors:
            file_path = os.path.join(self.path, f, "custodian.json")
            with open(file_path, "r") as file:
                error_dict[f] = json.load(file)

        volume_list = []
        for elec_folder in elec_folders:
            volume = extract_volume(
                os.path.join(self.path, elec_folder, "POSCAR.elec_dos")
            )
            volume_list.append(volume)

        poscar_list = []
        for elec_folder in elec_folders:
            poscar = Poscar.from_file(
                os.path.join(self.path, elec_folder, "POSCAR.elec_dos")
            )
            poscar_list.append(poscar.as_dict())

        thermal_electronic_properties_fit_reset = (
            thermal_electronic_properties_fit.reset_index()
        )
        min_temperature = (
            thermal_electronic_properties_fit_reset["temperature"].min().item()
        )
        max_temperature = (
            thermal_electronic_properties_fit_reset["temperature"].max().item()
        )
        dT = (
            thermal_electronic_properties_fit_reset["temperature"]
            .diff()
            .unique()[1]
            .item()
        )

        volume_range = thermal_electronic_properties_fit_reset["volume_fit"].to_list()[
            0
        ]
        min_volume = min(volume_range)
        max_volume = max(volume_range)

        polynomial_columns = [
            "f_el_poly",
            "s_el_poly",
            "cv_el_poly",
        ]
        polynomial_data = thermal_electronic_properties_fit_reset[polynomial_columns]
        f_el_coefficients = polynomial_data["f_el_poly"].to_list()
        s_el_coefficients = polynomial_data["s_el_poly"].to_list()
        cv_el_coefficients = polynomial_data["cv_el_poly"].to_list()

        for i in range(len(f_el_coefficients)):
            f_el_coefficients[i] = f_el_coefficients[i].coefficients.tolist()
            s_el_coefficients[i] = s_el_coefficients[i].coefficients.tolist()
            cv_el_coefficients[i] = cv_el_coefficients[i].coefficients.tolist()

        fel_data = {
            "$set": {
                "properties.Fel": {
                    "electronic DOS": {
                        "natoms elec": natoms_elec,
                        "VASP input": {
                            "INCAR elec": incar_elec_dos,
                            "KPOINTS": kpoints.as_dict(),
                            "POTCAR": potcar.as_dict(),
                            "errors": error_dict,
                        },
                        "volume": volume_list,
                        "POSCAR": poscar_list,
                        "temperature": {
                            "min": min_temperature,
                            "max": max_temperature,
                            "dT": dT,
                        },
                        "volume range": {"min": min_volume, "max": max_volume},
                        "Fel coefficients": f_el_coefficients,
                        "Sel coefficients": s_el_coefficients,
                        "Cv el coefficients": cv_el_coefficients,
                    },
                }
            }
        }

        document_id = ObjectId(object_id)
        self.collection.update_one({"_id": document_id}, fel_data)

        return fel_data

    def store_qha(
        self,
        object_id: str,
        quasi_harmonic_properties: pd.DataFrame,
        doc_name: str,
        include_tec: bool,
        eos: str,
    ):

        natoms = quasi_harmonic_properties["number_of_atoms"].unique()[0].item()
        pressure = quasi_harmonic_properties["pressure"].unique()[0].item()

        temperature = quasi_harmonic_properties["temperature"].values
        min_temperature = temperature.min().item()
        max_temperature = temperature.max().item()
        dT = (temperature[1] - temperature[0]).item()

        volume_range = quasi_harmonic_properties["volume_range"].values[0]
        min_volume = volume_range.min().item()
        max_volume = volume_range.max().item()

        V0_list = quasi_harmonic_properties["V0"].values.tolist()
        G0_list = quasi_harmonic_properties["G0"].values.tolist()
        B_list = quasi_harmonic_properties["B"].values.tolist()
        S0_list = quasi_harmonic_properties["S0"].values.tolist()
        H0_list = quasi_harmonic_properties["H0"].values.tolist()
        CTE_list = quasi_harmonic_properties["CTE"].values.tolist()
        Cp_list = quasi_harmonic_properties["Cp"].values.tolist()

        eos_constants = quasi_harmonic_properties["eos_constants"].tolist()
        eos_dict = []
        for i in eos_constants:
            eos_dict.append({"a": i[0], "b": i[1], "c": i[2], "d": i[3], "e": i[4]})

        qha_data = {
            "$set": {
                f"properties.QHA {doc_name}": {
                    "TEC": include_tec,
                    "natoms": natoms,
                    "pressure": pressure,
                    "temperature": {
                        "min": min_temperature,
                        "max": max_temperature,
                        "dT": dT,
                    },
                    "volume range": {"min": min_volume, "max": max_volume},
                    "V0": V0_list,
                    "G0": G0_list,
                    "B": B_list,
                    "S0": S0_list,
                    "H0": H0_list,
                    "CTE": CTE_list,
                    "Cp": Cp_list,
                    "EOS fit": {"EOS": eos, "constants": eos_dict},
                }
            }
        }

        document_id = ObjectId(object_id)
        self.collection.update_one({"_id": document_id}, qha_data)

    def store_expt(self, object_id: str, expt_properties: list[dict]):

        expt_data = {
            "$set": {
                "properties.experiments": expt_properties,
            }
        }

        document_id = ObjectId(object_id)
        self.collection.update_one({"_id": document_id}, expt_data)
