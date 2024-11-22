# Modify the following script for ASE VASP interface: 'run_vasp.py'
def generate_vasp_script():
    return """import ase, ase.io
import ase.io.vasp
from ase.calculators.vasp import Vasp
import numpy as np
import scipy.linalg as LA
import sys, os

# generate Kpoint grids from kspacing value
def kpoint_grid(cell, dx):
    rc = cell.cell.reciprocal()
    l_rc = np.empty(3, dtype=np.float64)
    for i in range(3):
        l_rc[i] = LA.norm(rc[i], 2)
    v = l_rc[:]/dx

    k_grid_rounded = list(map(int, np.rint(v)))
    for i in range(3):
        if k_grid_rounded[i] == 0:
            k_grid_rounded[i] = 1
    return k_grid_rounded

# setup VASP calculator
calc = Vasp(
     txt=      'stdout',
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
     kpar=5,
     ncore=5,
     )

# read input structure
atom = ase.io.read("POSCAR")

# if structure too large, set kpoints gamma only.
if len(atom) > 300:
    kpts = [1,1,1]
else:
    kpts = kpoint_grid(atom, 0.03)

# for slab calculation, set kpoints for z to 1.
if kpts[2] != 1:
    kpts[2] = 1
print("k-points: ", kpts)
calc.set(kpts=kpts)

atom.calc = calc

atom.get_potential_energy()

# in case SCF was not converged within NELM
file_path = 'stdout'
with open(file_path, 'r') as file:
    file_contents = file.read()

if 'NELM' in file_contents:
    os.system("touch NELM")
    calc.set(ismear=1)
    calc.set(sigma=0.2)
    atom.calc = calc
    atom.get_potential_energy()
    with open(file_path, 'r') as file:
        file_contents = file.read()
    if 'NELM' in file_contents:
        os.system("touch NELM")
        os.system("qsub control.cmd")
    else:
        os.system("rm WAVECAR vasprun.xml NELM")
else:
    os.system("rm WAVECAR vasprun.xml NELM")
"""

