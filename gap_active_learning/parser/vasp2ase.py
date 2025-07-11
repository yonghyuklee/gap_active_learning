import ase, ase.io
import re, os
import copy
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
from ase.units import bar, eV, Angstrom
from random import randrange
from gap_active_learning.parser.xyz2data import xyz2data

# Compile regular expressions for parsing relevant sections of the output
re_forces = re.compile(r"^\s*POSITION")            # Match lines starting with "POSITION"
re_virial = re.compile(r"^\s*FORCE on cell")       # Match lines starting with "FORCE on cell"
re_ewald = re.compile(r"^\s*electron-ion")         # Match lines starting with "electron-ion"
re_cell = re.compile(r"^\s*direct lattice vectors") # Match lines starting with "direct lattice vectors"
re_energy = re.compile(r"^\s*energy  without entropy")  # Match lines starting with "energy without entropy"
re_temperature = re.compile(r"^\s*kin\. lattice")  # Match lines starting with "kin. lattice"
re_energies = re.compile(r"^\s*Step ")            # Match lines starting with "Step "
re_magmom = re.compile(r"^\s*magnetization \(x\)")            # Match lines starting with "Step "

kbar2eV_angstrom3 = bar*1000/eV/Angstrom**3

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def parse_forces_energy(fd, check_virials=True, check_magmoms=True):
    line = fd.readline()
    forces = []
    energies = []
    # stresses = []
    virials = []
    magmoms = []
    while line:
        if re_forces.match(line):
            force = []
            line = fd.readline()
            line = fd.readline()
            w = line.split()
            while len(w) == 6:
                force.append(np.array(w[3:],dtype=float))
                line = fd.readline()
                w = line.split()
            forces.append(force)
        elif re_energy.match(line):
            e = float(line.split()[6])
            energies.append(e)
        elif check_virials and re_virial.match(line):
            while line:
                if line.startswith("  Total"):
                    break
                line = fd.readline()
            w = line.split()
            # Extract the relevant stress tensor values and convert them
            tensor = [float(w[i]) for i in [1, 4, 6, 4, 2, 5, 6, 5, 3]]
            # Store virials
            virials.append(" ".join(map(str, tensor)))
            virials.append("{} {} {} {} {} {} {} {} {}".format(w[1], w[4], w[6], w[4], w[2], w[5], w[6], w[5], w[3]))
            # Store the converted tensor in a structured format (e.g., as a tuple or string)
            # tensor = [float(w[i]) / atoms.get_volume() for i in [1, 4, 6, 4, 2, 5, 6, 5, 3]]
            # stresses.append(" ".join(map(str, tensor)))
        elif check_magmoms and re_magmom.match(line):
            magmom = []
            while line:
                if line.startswith("------------------------------------------"):
                    break
                line = fd.readline()
            line = fd.readline()
            w = line.split()
            while len(w) == 5:
                magmom.append(float(w[-1]))
                line = fd.readline()
                w = line.split()
            magmoms.append(magmom)
        line = fd.readline()
    forces = np.array(forces)
    magmoms = np.array(magmoms)
    return forces,energies,virials,magmoms #,stresses

def parse_structure(atom, fd):
    line = fd.readline()
    atoms = []
    temperature = []
    noa = len(atom)
    asymb = atom.get_chemical_symbols()
    while line:
        if re_temperature.match(line):
            temperature.append(float(line.split()[5]))
        if re_cell.match(line):
            cell = []
            for dim in range(3):
                line = fd.readline()
                line_split = line.split()[0:3]
                if all([is_float(i) for i in line_split]) is True:
                    cell.append([float(x) for x in line.split()[0:3]])
                else:
                    # OUTCAR sometimes prints lattice parameters without space 
                    # if one of the lattice vector elements is negative and larger than 10. 
                    # (e.g., 11.138247495-10.479611738)
                    cell_temp = []
                    for a in line_split:
                        if '-' in a and a.startswith('-'):
                            i = ['-'+e for e in a.split('-') if e]
                            for a in i:
                                cell_temp.append(a)
                        elif '-' in a and not a.startswith('-'):
                            cell_temp.append(a.split('-')[0])
                            i = ['-'+e for e in a.split('-')[1:] if e]
                            for a in i:
                                cell_temp.append(a)
                    cell.append(cell_temp[0:3])
                    
        if re_forces.match(line):
            elements = []
            pos = []
            line = fd.readline()
            line = fd.readline()
            w = line.split()
            for k in range(noa):
                pos.append(np.array(w[0:3],dtype=float))
                elements.append(asymb[k])
                line = fd.readline()
                w = line.split()
            atoms.append(ase.Atoms(elements,positions=pos,pbc=True,cell=cell))
        line = fd.readline()
    ase.io.write('vasp_trajectory.xyz', atoms)
    return atoms, cell, temperature


def process_ase(
                initial_structure,
                dft_output,
                check_virials=True,
                check_magmoms=True,
                ):
    snap1 = copy.deepcopy(initial_structure)
    try:
        del snap1.info['dft_energy']
    except:
        pass
    traj, cell, temp = parse_structure(snap1, open(dft_output,'r'))
    snaps = traj[:-1]
    if snap1.cell.sum != 0:
        snap1.pbc = True
        snap1.cell = cell
    atoms = [snap1]
    atoms = atoms + snaps
    dft_forces, dft_energies, dft_virials, dft_magmoms = parse_forces_energy(open(dft_output,'r'), check_virials=check_virials, check_magmoms=check_magmoms)
    for n, s in enumerate(atoms):
        if len(temp) > 0:
            s.info['temperature'] = temp[n]
        s.info['dft_energy'] = dft_energies[n]
        if check_virials:
            s.info['dft_virial'] = dft_virials[n]
        s.set_array('dft_forces',dft_forces[n])
        if check_magmoms:
            s.set_array('dft_magmoms',dft_magmoms[n])
    return atoms, dft_energies


def plot_energy(
                trajectory_name,
                key = 'dft_energy',
                filename = 'energy_vs_snaps',
                ):
    trajectory = ase.io.read(trajectory_name,':')
    energies = []
    for s in trajectory:
        energies.append(s.info[key])
    plt.plot(energies)
    plt.xlabel('Snapshots')
    plt.ylabel('Energy [eV]')
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--folders', type=str, nargs='+', default=[os.getcwd()],
                        help='Provide folders that contain initial structure and OUTCAR (VASP) file')
    parser.add_argument('-sn', '--structure_name', type=str, default='POSCAR',
                        help='Name of the input structure (default: POSCAR)')
    parser.add_argument('-on', '--output_name', type=str, default='OUTCAR',
                        help='Name of the output structure (default: OUTCAR)')
    parser.add_argument('-ol', '--only_last_snapshot', action='store_true',
                        help='Only take the last snapshot of each structure')
    parser.add_argument('-of', '--only_first_snapshot', action='store_true',
                        help='Only take the first snapshot of each structure')
    parser.add_argument('-olf', '--only_last_and_first_snapshot', action='store_true',
                        help='Only take the last+first snapshot of each structure')
    parser.add_argument('-addinfo', '--addinfo', action='store_true',
                        help='Take folder name as info for info_string')
    parser.add_argument('-20','--twenty',action='store_true',
                        help='Only take first and 20th snapshot of each structure')
    parser.add_argument('-n','--nth',
                        help='take nth snapshot of each structure')
    parser.add_argument('-r','--rand',action='store_true',
                        help='take random snapshot of each structure')
    parser.add_argument('-v','--check_virials',action='store_false',default=True,
                        help='NOT extract virials')
    parser.add_argument('-m','--check_magmoms',action='store_false',default=True,
                        help='NOT extract magnetic moments')
    parser.add_argument('-name','--name', type=str,
                        help='Name of structure info',)
    parser.add_argument('-fn', '--file_name', type=str, default='force_trajectory.xyz',
                        help='Name of the output file')

    args = parser.parse_args()
    trajectory = []

    sn = args.structure_name
    on = args.output_name
    fn = args.file_name

    cv = args.check_virials
    cm = args.check_magmoms


    for f in args.folders:
        s = ase.io.read('{}/{}'.format(f, sn))
        result = process_ase(s,
                             '{}/{}'.format(f,on),
                             check_virials=cv,
                             check_magmoms=cm,
                             )
        s = result[0]
        for k in s:
            k.info['structure'] = 'slab'
            if args.name:
                k.info['structure_info'] = args.name
            else:
                k.info['structure_info'] = '-'.join(f.split('/')[-4:])
        if args.only_last_snapshot:
            s = [s[-1]]
        elif args.only_first_snapshot:
            s = [s[0]]
        elif args.only_last_and_first_snapshot:
            print(len(s))
            if len(s) > 1:
                s = [s[0], s[-1]]
        elif args.nth:
            print(len(s))
            if len(s) > int(args.nth):
                s = [s[int(args.nth)]]
        elif args.rand:
            n = randrange(len(s))
            print(n)
            s = [s[n-1]]
        trajectory  += s
    ase.io.write(fn, trajectory)
    ase.io.write('final.in', trajectory[-1])
    plot_energy(fn)
    
    try:
        os.remove('vasp_trajectory.xyz')
    except:
        pass
