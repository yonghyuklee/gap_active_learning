import ase, ase.io
import ase.io.espresso
import re, os
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
from ase.units import Ry, eV, Bohr, Angstrom

Ry2Ev = Ry/eV
bohr2AA = Bohr/Angstrom
Ry_bohr2Ev_AA = Ry2Ev / bohr2AA

re_forces    = re.compile(r"""^\s*Forces acting on atoms""")
re_positions = re.compile(r"""^\s*ATOMIC_POSITIONS""")
re_celldim = re.compile(r"""^\s*celldm\(1""")
re_cell = re.compile(r"""^\s*crystal axes\:""")
re_CELL = re.compile(r"""^CELL_PARAMETERS""")
re_energy = re.compile(r"""^\s*!    total energy""")
re_noa = re.compile(r"""^\s*     number of atoms""")
re_temperatur = re.compile(r"""^\s*    Ekin =""")

def parse_forces_energy(fd):
    line = fd.readline()
    forces = []
    energies = []
    flist = []
    n = 0
    while line:
        if re_forces.match(line):
            force = []
            line = fd.readline()
            line = fd.readline()
            w = line.split()
            while w[0] == 'atom':
                try:
                    force.append(np.array(w[6:],dtype=float))
                except:
                    flist.append(n)
                    break
                line = fd.readline()
                w = line.split()
            else:
                forces.append(force)
            n += 1
        elif re_energy.match(line):
            e = float(line.split()[-2]) * Ry2Ev
            energies.append(e)
        line = fd.readline()
    forces = np.array(forces)
    for i in sorted(flist, reverse=True):
        del energies[i]
    forces *= Ry_bohr2Ev_AA
    return forces,energies,flist

def parse_structure(fd):
    line = fd.readline()
    atoms = []
    temperature = []
    while line:
        if re_temperatur.match(line):
            temperature.append(float(line.split()[6]))
        if re_celldim.match(line):
            w = line.split()
            try:
                celldim = float(w[1])
            except:
                celldim = float(w[2])
        elif re_noa.match(line):
            noa = int(line.split()[-1])
        elif re_cell.match(line):
            cell = []
            for dim in range(3):
                line = fd.readline()
                cell.append([float(x) for x in line.split()[-4:-1]])
            cell = np.array(cell) * celldim * 0.529177
        elif re_CELL.match(line):
            try:
                alat = float(line.split()[-1].split(')')[0])* 0.529177
            except:
                alat = 1.
            CELL = []
            CELL.append([alat*float(x) for x in fd.readline().split()])
            CELL.append([alat*float(x) for x in fd.readline().split()])
            CELL.append([alat*float(x) for x in fd.readline().split()])
        if re_positions.match(line):
            elements = []
            pos = []
            line = fd.readline()
            w = line.split()
            try:
                for k in range(noa):
                    pos.append(np.array(w[1:4],dtype=float))
                    elements.append(w[0])
                    line = fd.readline()
                    w = line.split()
            except:
                while len(w) == 4:
                    pos.append(np.array(w[1:],dtype=float))
                    elements.append(w[0])
                    line = fd.readline()
                    w = line.split()
            try:
                atoms.append(ase.Atoms(elements,positions=pos,pbc=True,cell=CELL))
            except:
                atoms.append(ase.Atoms(elements,positions=pos,pbc=True,cell=cell))
        line = fd.readline()
    ase.io.write('qe_trajectory.xyz',atoms)
    return atoms,cell,temperature

def process_ase(
                initial_structure,
                dft_output,
                ):
    snap1 = copy.deepcopy(initial_structure)
    try:
        del snap1.info['dft_energy']
    except:
        pass
    traj, cell, temp = parse_structure(open(dft_output,'r'))
    snaps = traj[:-1]
    if snap1.cell.sum != 0:
        snap1.pbc = True
        snap1.cell = cell
    atoms = [snap1]
    atoms = atoms + snaps
    dft_forces, dft_energies, flist = parse_forces_energy(open(dft_output,'r'))
    for i in sorted(flist, reverse=True):
        del atoms[i]
        if len(temp) > 0:
            del temp[i]
    for n,s in enumerate(atoms):
        if len(temp) > 0:
            s.info['temperature'] = temp[n]
        s.info['dft_energy'] = dft_energies[n]
        s.set_array('dft_forces',dft_forces[n])
        try:
            del s.info['comment']
        except:
            pass
    return atoms, dft_energies


def clean_up(
             traj,
             ):
    fd = open(traj,'r+')
    of = open('temp.xyz','w')
    n = 0
    for line in fd.readlines():
        line = str.replace(line, '=T',' ')
        of.write(line)
    os.rename('temp.xyz',traj)

def include_relative_energies(
                              traj_name,
                              atomic_energies = {
                                                'Bi' : -1846.5752131181798,
                                                'V'  : -1929.5551813821444,
                                                'O'  : -429.8317508590434,
                                                 },
                              ):
    traj = ase.io.read(traj_name,':')
    for s in traj:
        try:
            w = s.info['comment'].split('=')
            s.info[w[0]] = float(w[1])
            del s.info['comment']
        except:
            pass
        aten = 0
        cs = s.get_chemical_symbols()
        for el,en in atomic_energies.items():
           noa = sum(np.array(cs) == el)
           aten += noa*en
        s.info['relative_energy'] = ( s.info['dft_energy'] - aten ) / len(s)
    ase.io.write(traj_name, traj)


def plot_energy(
                trajectory_name,
                key = 'dft_energy',
                filename = 'energy_vs_snaps',
                ):
    trajectory = ase.io.read(trajectory_name,':')
    energies = []
    for s in trajectory:
        energies.append(s.info['dft_energy'])
    plt.plot(energies)
    plt.xlabel('Snapshots')
    plt.ylabel('Energy [eV]')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f','--folders',type=str, nargs='+',default=[os.getcwd()],
                        help='Provide folders that contain structure.xyz and out (QE) file')

    parser.add_argument('-sn','--structure_name',
                        type=str,
                        default='structure.xyz',
                        help='Name of the input structure e.g. structure.xyz or geometry.in'
                        )

    parser.add_argument('-on','--output_name',
                        type=str,
                        default='out',
                        help='Name of the QE output file',
                        )

    parser.add_argument('-ol','--only_last_snapshot',action='store_true',
                        help='Only take last snapshot of each structure (not the whole MD)',
                        )
    parser.add_argument('-of','--only_first_snapshot',action='store_true',
                        help='Only take first snapshot of each structure (not the whole MD)',
                        )
    parser.add_argument('-olf','--only_last_and_first_snapshot',action='store_true',
                        help='Only take last+first snapshot of each structure (not the whole MD)',
                        )
    parser.add_argument('-addinfo','--addinfo',action='store_true',
                        help='take folder name as info for info_string',
                        )
    parser.add_argument('-20','--twenty',action='store_true',
                        help='Only take first and 20th snapshot of each structure (not the whole MD)',
                        )
    parser.add_argument('-name','--name',type=str,
                        help='Name of structure info',
                        )


    args = parser.parse_args()
    trajectory = []

    sn = args.structure_name
    on = args.output_name

    for f in args.folders:
        try:
            s = ase.io.read('%s/forces.xyz'%f,':')
            for k in s:
                k.info['structure'] = 'slab'
                k.info['structure_info'] = '-'.join(f.split('/'))
            if args.only_last_snapshot:
                s = [s[-1]]
            elif args.only_first_snapshot:
                s = [s[0]]
            elif args.only_last_and_first_snapshot:
                print(len(s))
                if len(s) > 1:
                    s = [s[0],s[-1]]
            elif args.twenty:
                print(len(s))
                s = [s[0],s[19]]
            if args.addinfo:
                try:
                    info_string = f.split('/')
                    info_string = '-'.join(info_string[:2])
                    for ss in s:
                        ss.info['structure_info'] = info_string
                    print(s[0].info['structure_info'])
                except:
                    pass
            trajectory += s
            print('Taking %s/forces.xyz'%f)
        except:
            print('No forces.xyz file found, extracting forces from %s/out'%f)
            try:
                s = ase.io.espresso.read_espresso_in('%s/%s'%(f,sn))
            except:
                s = ase.io.read('%s/%s'%(f,sn))
            result = process_ase(s,
                                 '%s/%s'%(f,on),
                                 )
            s = result[0]
            if args.name:
                for k in s:
                    k.info['structure'] = 'slab'
                    k.info['structure_info'] = args.name
            else:
                for k in s:
                    k.info['structure'] = 'slab'
                    k.info['structure_info'] = '-'.join(f.split('/'))
            if args.only_last_snapshot:
                s = [s[-1]]
            elif args.only_last_and_first_snapshot:
                print(len(s))
                if len(s) > 1:
                    s = [s[0],s[-1]]
            trajectory += s
    ase.io.write('force_trajectory.xyz',trajectory)
    ase.io.write('final.in',trajectory[-1])
    clean_up('force_trajectory.xyz')
    include_relative_energies('force_trajectory.xyz')
    plot_energy('force_trajectory.xyz')

    try:
        os.remove('qe_trajectory.xyz')
    except:
        pass

    try:
        os.remove('log.lammps')
        os.remove('dump.dump')
        os.remove('tmp.data')
        os.remove('temp')
    except:
        pass
