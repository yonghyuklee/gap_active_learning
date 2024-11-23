# Modify the following script for ASE VASP interface: 'run_vasp.py'
def generate_vasp_script():
    return """import ase
import ase.io
from ase.calculators.vasp import Vasp
import numpy as np
import os
import logging
from itertools import cycle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
INPUT_FILE = "POSCAR"
STDOUT_FILE = "stdout"
NELM_MARKER = "NELM"
MAX_ITERATIONS = 10

def kpoint_grid(cell: ase.Atoms, dx: float) -> list[int]:
    \"\"\"
    Generate K-point grids from K-spacing value.
    \"\"\"
    rc = cell.cell.reciprocal()
    l_rc = np.linalg.norm(rc, axis=1)
    k_grid_rounded = list(map(int, np.rint(l_rc / dx)))
    return [max(1, val) for val in k_grid_rounded]

def check_nelm_in_file(file_path: str) -> bool:
    \"\"\"
    Check if the string 'NELM' exists in the content of a given file.
    \"\"\"
    try:
        with open(file_path, 'r') as file:
            return any('NELM' in line.upper() for line in file)
    except FileNotFoundError:
        logging.error(f"The file '{file_path}' was not found.")
        return False
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}")
        return False

# Check input structure
if not os.path.exists(INPUT_FILE):
    logging.error(f"Input structure file '{INPUT_FILE}' not found. Exiting.")
    sys.exit(1)

# Read input structure
atom = ase.io.read(INPUT_FILE)

# Setup VASP calculators
calc = Vasp(
     txt=      STDOUT_FILE,
     prec=     'Accurate',
     xc=       'PBE',
     pp=       'PBE',
     gamma=    True,
     reciprocal= True,
     encut=    500,
     ediff=    1E-5,
     ismear=   0,
     sigma=    0.05,
     nelm=     200,
     nsw=      0,
     ispin=    2,
     ibrion=   -1,
     lwave=    True,
     lcharg=   False,
     lreal=    False,
     lasph=    True,
     lorbit=   10,
     ivdw=     20,
     setups=   {'base': 'recommended'},
     ldau_luj= {'Ti': {'L': 2, 'U': 3.5, 'J': 0},
                'O':  {'L': -1, 'U': 0, 'J': 0},
                'H':  {'L': -1, 'U': 0, 'J': 0},
                'Rh': {'L': -1, 'U': 0, 'J': 0},
                },
     lmaxmix=  4,
     ldauprint=0,
     ldautype= 2,
     amix=     0.1,
     bmix=     0.0001,
     kpar=     5,
     ncore=    5,
     )

# Adjust k-points
kpts = kpoint_grid(atom, 0.03) if len(atom) <= 300 else [1, 1, 1]
kpts[2] = 1
calc.set(kpts=kpts)
logging.info(f"K-points set to: {kpts}")

# Create fallback calculator
calc2 = calc.copy()
calc2.set(imix=1)
calcs = [calc, calc2]
calc_cycle = cycle(calcs)

# SCF loop
finished = False
iteration = 0

while not finished and iteration < MAX_ITERATIONS:
    current_calc = next(calc_cycle)
    atom.calc = current_calc

    try:
        atom.get_potential_energy()
        logging.info(f"SCF calculation performed. Iteration: {iteration + 1}")
    except Exception as e:
        logging.error(f"Error during SCF calculation: {e}")
        break

    if check_nelm_in_file(STDOUT_FILE):
        logging.warning("SCF not converged. Retrying...")
        os.system(f"touch {NELM_MARKER}")
    else:
        logging.info("SCF converged successfully.")
        os.system(f"rm -f WAVECAR vasprun.xml {NELM_MARKER}")
        finished = True

    iteration += 1

if not finished:
    logging.error("SCF did not converge within the maximum allowed iterations.")

"""