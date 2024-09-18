import ase, ase.io
import re, os, sys
import copy
import string
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
from random import randrange
from gap_active_learning.parser.xyz2data import xyz2data

re_forces = re.compile("""^\s*POSITION""")
re_virial = re.compile("""^\s*FORCE on cell""")
re_ewald = re.compile("""^\s*electron-ion""")
re_cell = re.compile("""^\s*direct lattice vectors""")
re_energy = re.compile("""^\s*energy  without entropy""")
re_temperature = re.compile("""^\s*kin. lattice""")
re_energies = re.compile("""^\s*Step """)

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def parse_forces_energy(fd):
    line = fd.readline()
    # ewalds = []
    forces = []
    energies = []
    virials = []
    while line:
        # if re_ewald.match(line):
        #     ewald = []
        #     line = fd.readline()
        #     line = fd.readline()
        #     w = line.split()
        #     while len(w) == 12:
        #         ewald.append(np.array(w[3:6],dtype=float))
        #         line = fd.readline()
        #         w = line.split()
        # ewalds.append(ewald)
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
        elif re_virial.match(line):
            # print(line)
            for _ in range(14):
                line = fd.readline()
            w = line.split()
            virials.append("{} {} {} {} {} {} {} {} {}".format(w[1], w[4], w[6], w[4], w[2], w[5], w[6], w[5], w[3]))
        line = fd.readline()
    # ewalds = np.array(ewalds)
    forces = np.array(forces)
    # for e in ewalds[0]:
    #     print(e)
    return forces,energies,virials #,ewalds

def parse_structure(atom, fd):
    line = fd.readline()
    atoms = []
    temperature = []
    # atom = ase.io.read("POSCAR")
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

def parse_dump(
               structure,
               ):
    atoms = ase.io.read(structure, ":")
    f = atoms[0].get_forces()
    c = atoms[0].arrays['initial_charges']
    d = pd.DataFrame()
    d = pd.concat([d, pd.DataFrame(f)], ignore_index=True)
    d = d.rename(columns={0: 'fx', 1: 'fy', 2: 'fz'})
    d['q'] = c
    return d

def lammps_energy(
                  logfile,
                  ):
    lf = open(logfile,'r')
    line = lf.readline()
    energies = []
    while line:
        if re_energies.match(line):
            w = line.split()
            n = np.where(np.array(w) == 'TotEng')[0][0]
            line = lf.readline()
            while line: 
                try:
                    w = line.split()
                    energies.append([float(a) for a in line.split()][n])
                except:
                    break
                line = lf.readline() 
        line = lf.readline()
    return energies

def QEqforces_energies(
                       structure,
                       mpi=True,
                       basedir = '/Users/yonghyuk/Dropbox/1_Project/CuPd_Zirconia/script/basedir/'
                       ):
    s = copy.deepcopy(structure)
    vacuum_layer = 0
    slab = True
    if s.cell.sum() == 0:
        vacuum_layer = 10
        slab = False
    xyz2data(s,
             vacuum_layer = vacuum_layer, 
             filename = 'tmp.data',
             slab = slab,
             qO = -0.74,
             )
    if mpi:
        os.system('mpirun -np 4 /opt/homebrew/bin/lmp_mpi < %s/pureQEq.in > temp'%basedir)
    else:
        os.system('/opt/homebrew/bin/lmp_serial < %s/pureQEq.in > temp'%basedir)
    lq = parse_dump('dump.dump')
    le = lammps_energy('log.lammps')[0]   # Taking the energy of the first step 
    return lq, le

def FixedQ_forces_energies(
                           structure,
                           mpi=True,
                           basedir = '/Users/yonghyuk/Dropbox/1_Project/CuPd_Zirconia/script/basedir/',
                           qO = -0.74,
                           ):
    s = copy.deepcopy(structure)
    vacuum_layer = 0
    slab = True
    if s.cell.sum() == 0:
        vacuum_layer = 10
        slab = False
    xyz2data(s,
             vacuum_layer = vacuum_layer, 
             filename = 'tmp.data',
             slab = slab,
             qO = qO,
             )
    if mpi:
        os.system('mpirun -np 4 /opt/homebrew/bin/lmp_mpi < %s/FixedCharge.in > temp'%basedir)
    else:
        os.system('/opt/homebrew/bin/lmp_serial < %s/FixedCharge.in > temp'%basedir)
    lq = parse_dump('dump.dump')
    le = lammps_energy('log.lammps')[0]   # Taking the energy of the first step 
    return lq, le

def process_ase(
                initial_structure,
                dft_output,
                qeq = True,
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
    # dft_forces, dft_energies, dft_ewalds = parse_forces_energy(open(dft_output,'r'))
    dft_forces, dft_energies, dft_virials = parse_forces_energy(open(dft_output,'r'))
    for n, s in enumerate(atoms):
        if len(temp) > 0:
            s.info['temperature'] = temp[n]
        s.info['dft_energy'] = dft_energies[n]
        s.info['dft_virial'] = dft_virials[n]
        if qeq:
            lq, le = QEqforces_energies(s)
            qeq_forces = lq[['fx','fy','fz']].values
            s.set_array('qeq_forces',qeq_forces)
            s.set_array('qeq_charges',lq.q.values)
            s.info['qeq_energy'] = le
            s.set_array('diff_forces',dft_forces[n]-qeq_forces)

            lq, le = FixedQ_forces_energies(s)
            fq_forces = lq[['fx','fy','fz']].values
            s.set_array('fq_forces',fq_forces)
            s.set_array('fq_charges',lq.q.values)
            s.info['fq_energy'] = le
            s.set_array('fq_diff_forces',dft_forces[n]-fq_forces)
        s.set_array('dft_forces',dft_forces[n])
        # s.set_array('dft_ewalds',dft_ewalds[n])
    return atoms, dft_energies

def update_charges(
                   traj_name,
                   qeq = False,
                   fc  = False,
                   qO  = -0.74,
                   ):
    traj = ase.io.read(traj_name,':')
    t = []
    for n,s in enumerate(traj):
        dft_forces = s.arrays['dft_forces'] 
        if qeq:
            lq, le = QEqforces_energies(s)
            qeq_forces = lq[['fx','fy','fz']].values
            s.set_array('qeq_forces',qeq_forces)
            s.set_array('qeq_charges',lq.q.values)
            s.info['qeq_energy'] = le
            s.set_array('diff_forces',dft_forces-qeq_forces)
        if fc:
            lq, le = FixedQ_forces_energies(
                                            s,
                                            qO = qO,
                                            )
            fq_forces = lq[['fx','fy','fz']].values
            s.set_array('fq_forces',fq_forces)
            s.set_array('fq_charges',lq.q.values)
            s.info['fq_energy'] = le
            s.set_array('fq_diff_forces',dft_forces-fq_forces)
        t.append(s)
    return t

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

def include_energy_difference(
                              traj_name,
                              qeq = True,
                              ):
    fin = open(traj_name,'r') 
    traj = ase.io.read(traj_name,':')
    fout = open('temp.xyz','w')
    for n,s in enumerate(traj):
        s.wrap()
        try:
            w = s.info['comment'].split('=')
            s.info[w[0]] = float(w[1])
            del s.info['comment']
        except:
            pass
        noa = s.get_global_number_of_atoms()
        dft_te = s.info['dft_energy']
        if qeq:

            qeq_te = s.info['qeq_energy'] 
            fq_te = s.info['fq_energy']
            diff_te = dft_te - qeq_te
            fq_diff_te = dft_te - fq_te

            fin.readline()
            fout.write('%s\n'%noa)
            line = fin.readline()
            line = line.rstrip() + ' diff_energy=%s\n'%diff_te
            line = line.rstrip() + ' fq_diff_energy=%s\n'%fq_diff_te
            fout.write(line)
        for a in range(noa):
            l = fin.readline()
            fout.write(l)
    fout.close()
    os.rename('temp.xyz',traj_name)
#    traj = ase.io.read(traj_name,':')
#    ase.io.write(traj_name,traj)

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
    parser.add_argument('-nq','--no_qeq',action='store_false',
                        help='Exclude QEq forces calculation')
    parser.add_argument('-qO','--qO', type=float, default=-0.74,
                        help='Oxygen charge applied to calculate coulomb for fixed charge model',)
    parser.add_argument('-addinfo', '--addinfo', action='store_true',
                        help='Take folder name as info for info_string')
    parser.add_argument('-20','--twenty',action='store_true',
                        help='Only take first and 20th snapshot of each structure')
    parser.add_argument('-n','--nth',
                        help='take nth snapshot of each structure')
    parser.add_argument('-r','--rand',action='store_true',
                        help='take random snapshot of each structure')
    parser.add_argument('-name','--name', type=str,
                        help='Name of structure info',)
    parser.add_argument('-nm','--no_mpi', action='store_false',
                        help='Not using mpi lammps version',)

    args = parser.parse_args()
    trajectory = []

    sn = args.structure_name
    on = args.output_name

    for f in args.folders:
        s = ase.io.read('{}/{}'.format(f, sn))
        result = process_ase(s,
                             '{}/{}'.format(f,on),
                             qeq = args.no_qeq,
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
    ase.io.write('force_trajectory.xyz', trajectory)
    ase.io.write('final.in', trajectory[-1])
    #include_energy_difference('force_trajectory.xyz',args.no_qeq)
    plot_energy('force_trajectory.xyz')
    
    try:
        os.remove('vasp_trajectory.xyz')
        os.remove('log.lammps')
        os.remove('dump.dump')
        os.remove('tmp.data')
        os.remove('temp')
    except:
        pass
