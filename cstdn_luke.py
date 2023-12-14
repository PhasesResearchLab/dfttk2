import os
import shutil

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler
from custodian.vasp.jobs import VaspJob
from pymatgen.core import structure
from pymatgen.io.vasp.outputs import Outcar, Vasprun

# Function to extract the last occurrence of volume from OUTCAR files
def extract_volume(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'volume' in line:
                volume = float(line.split()[-1])
                break  # Stop searching after finding the last occurrence
    return volume

# Function to extract the last occurrence of pressure from OUTCAR files
def extract_pressure(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'pressure' in line:
                pressure = float(line.split()[3])
                break  # Stop searching after finding the last occurrence
    return pressure

# Function to extract energy from OSZICAR files
def extract_energy(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in reversed(lines):
            if 'F=' in line:
                energy = float(line.split()[4])
                break  # Stop searching after finding the last occurrence
    return energy

def three_step_relaxation(path, vasp_cmd, handlers, backup=True): #path should contain necessary vasp config files
    orginal_dir = os.getcwd()
    os.chdir(path)
    step1 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = False,
    suffix = '.1relax',
    backup = backup,
            )
    
    step2 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = False,
    suffix = '.2relax',
    backup = backup,
    settings_override = [
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
            )
    
    step3 = VaspJob(
    vasp_cmd = vasp_cmd,
    copy_magmom = True,
    final = True,
    suffix = '.3static',
    backup = backup,
    settings_override = [
        {"dict": "INCAR", "action": {"_set": {
            "IBRION": -1,
        "NSW": 0,
            "ISMEAR": -5
            }}},
        {"file": "CONTCAR", "action": {"_file_copy": {"dest": "POSCAR"}}}
        ]
            )

    jobs = [step1, step2, step3]
    c = Custodian(handlers, jobs, max_errors = 3)
    c.run()
    os.chdir(orginal_dir)

def wavecar_prop_series(path, volumes, vasp_cmd, handlers): #path should contain starting POSCAR, POTCAR, INCAR, KPOINTS
    for i, vol in enumerate(volumes):
        #create vol folder
        vol_folder_name = 'vol_' + str(i)
        vol_folder_path = os.path.join(path, vol_folder_name)
        os.makedirs(vol_folder_path)

        if i == 0: #copy from path
            files_to_copy = ['INCAR', 'KPOINTS', 'POSCAR', 'POTCAR']
            for file_name in files_to_copy:
                if os.path.isfile(os.path.join(path, file_name)):
                    shutil.copy2(os.path.join(path, file_name), os.path.join(vol_folder_path, file_name))
        else: #copy from previous folder
            previous_vol_folder_path = os.path.join(path, 'vol_' + str(i-1))
            source_name_dest_name = [('CONTCAR.3static', 'POSCAR'),
                                ('INCAR.2relax', 'INCAR'),
                                ('KPOINTS.1relax', 'KPOINTS'),
                                ('POTCAR', 'POTCAR'),
                                ('WAVECAR.3static', 'WAVECAR'),
                                ('CHGCAR.3static', 'CHGCAR')]
            for file_name in source_name_dest_name:
                file_source = os.path.join(previous_vol_folder_path, file_name[0])
                file_dest = os.path.join(vol_folder_path, file_name[1])
                if os.path.isfile(file_source):
                    shutil.copy2(file_source, file_dest)  

        #change the volume of the POSCAR
        poscar = os.path.join(vol_folder_path, 'POSCAR')
        struct = structure.Structure.from_file(poscar)
        struct.scale_lattice(vol)
        struct.to_file(poscar, "POSCAR")
        
        #run vasp
        three_step_relaxation(vol_folder_path, vasp_cmd, handlers, backup=False)




if __name__ == "__main__":
    subset = list(VaspErrorHandler.error_msgs.keys())
    subset.remove("algo_tet")

    handlers = [VaspErrorHandler(errors_subset_to_catch = subset)]
    vasp_cmd = ["srun", "vasp_std"]

    wavecar_prop_series(os.getcwd(), [360, 350, 340, 330, 320, 310, 300, 290, 280, 270, 260], vasp_cmd, handlers)
