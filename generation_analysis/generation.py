import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys, argparse
import ase, ase.io
import ase.io.vasp
import pandas as pd
import tempfile
import pickle
import pathlib
import ase.io.espresso
import operator
import glob
from termcolor import colored

#from xyz2data import *
from vasp2ase import process_ase
#from lammps import *


import generation_analysis as ga


class GapGen:

    def __init__(
                 self,
                 gapfile,
                 trainingfile,
                 labels = [],
                 dft_files = {
                             'input_structure' : 'POSCAR',
                             'output' : 'OUTCAR',
                             'run_script': 'run_vasp.py',
                             },
                 md_files = {
                             'logfile' : 'log.lammps',
                             'final_structure' : 'final.in',
                             },
                 gcbh_files = {
                             'logfile' : 'grandcanonical.log',
                             'minima_structures' : 'local_minima.db'
                             },
                 md_dir = 'md/',

                 # NEED TO BE TESTED
                 kappa_min = 0.075,
                 uncertainty_min = 50,
                #  geoopt_maxsteps = 20,
                 max_selected = 100,

                 lammps_binding = {},
                 update = True,
                 ):

        if os.path.isfile(gapfile):
            self.gapfile = os.path.abspath(gapfile)
            self.gap_string = open(self.gapfile,'r').readline()[1:-2]
        else:
            print('Gapfile not found, initializing empty instance')
        if os.path.isfile(trainingfile):
            self.trainingfile = os.path.abspath(trainingfile)
            self.training_set = ase.io.read(os.path.abspath(trainingfile),':')
            if update:
                self.training_set = ga.similarity.update_structure(self.training_set)
            self.get_atomic_energies()
        else:
            raise IOError('training set not found')
        self.homedir = os.path.abspath('.')

        self.kappa_min = kappa_min
        self.uncertainty_min = uncertainty_min
        # self.geoopt_maxsteps = geoopt_maxsteps
        self.max_selected = max_selected

        self.mddir = os.path.join(self.homedir,md_dir)
        if lammps_binding:
            self.lammps_binding = lammps_binding
        else:
            lammps_binding = {}
            n = 1
            for s in self.training_set:
                if len(s) == 1:
                    lammps_binding[n] = s.get_atomic_numbers()[0]
                    n +=1
            self.lammps_binding = lammps_binding
            print('\nLammps binding extracted:')
        for k,v in lammps_binding.items():
            print('%s-%s'%(k,v))
        print('\n')

        if self.mddir[-1] != '/':
            self.mddir+= '/'
        self.dftdir = os.path.join(self.homedir,'dft/')
        self.dft_files = dft_files
        self.md_files = md_files
        self.gcbh_files = gcbh_files

        self.get_md_folders(
                            labels,
                            )
        self.get_gcbh_folders()

        print(self.folders)

    def get_md_folders(
                       self,
                       labels,
                       ):
        self.folders = {}
        # self.md_energies = pd.DataFrame()
        # initial_energies = []
        # final_energies = []
        if not labels:
            labels = []
        if len(labels) == 0:
            directories = []
            for root, dirs, _ in os.walk(self.mddir):
                # Exclude directories that start with 'opt'
                dirs[:] = [d for d in dirs if not d.startswith('opt')]
                directories.append(root)
                for subdir in dirs:
                    directories.append(os.path.join(root, subdir))
            for d in directories:
                logfile = os.path.join(d,self.md_files['logfile'])
                output_name = os.path.join(d,self.md_files['final_structure'])
                if os.path.isfile(logfile):
                    if self.lammps_finished(logfile):
                        self.write_final(logfile,output_name=output_name)
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        ie,fe = self.energy_lammps(logfile)
                        # final_energies.append(fe)
                        # initial_energies.append(ie)
                        labels.append(l)
            # self.md_energies['label'] = labels
            # self.md_energies['initial'] = initial_energies
            # self.md_energies['final'] = final_energies

    def get_gcbh_folders(
                       self,
                       ):
        if self.folders:
            pass
        else:
            self.folders = {}
        labels = []
        if len(labels) == 0:
            directories = []
            for root, dirs, _ in os.walk(self.mddir):
                # Exclude directories that start with 'opt'
                dirs[:] = [d for d in dirs if not d.startswith('opt')]
                directories.append(root)
                for subdir in dirs:
                    directories.append(os.path.join(root, subdir))
            for d in directories:
                logfile = os.path.join(d,self.gcbh_files['logfile'])
                # output_name = os.path.join(d,self.gcbh_files['minima_structures'])
                if os.path.isfile(logfile):
                    if self.gcbh_finished(logfile):
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        # ie,fe = self.energy_lammps(logfile)
                        # final_energies.append(fe)
                        # initial_energies.append(ie)
                        labels.append(l)
                    else:
                        print("GCBH is not converged!!")
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        # ie,fe = self.energy_lammps(logfile)
                        # final_energies.append(fe)
                        # initial_energies.append(ie)
                        labels.append(l)
            # self.md_energies['label'] = labels
            # self.md_energies['initial'] = initial_energies
            # self.md_energies['final'] = final_energies

    def get_dft_folders(
                        self,
                        ):
        self.dft_folders = {}
        for dinfo in os.walk(self.dftdir):
            d = dinfo[0]
            dft_output = os.path.join(d,self.dft_files['output'])
            if os.path.isfile(dft_output):
                if self.vasp_finished(dft_output):
                    path = os.path.abspath(d+'/../')
                    # print(path)
                    folder_extension = d.split('/')[-1]
                    l = path.replace(self.dftdir,'').replace('/','-')
                    if l in self.dft_folders:
                        self.dft_folders[l][folder_extension] = os.path.join(self.homedir,d)
                    else:
                        self.dft_folders[l] = {folder_extension : os.path.join(self.homedir,d)}


    def lammps_finished(
                        self,
                        logfile,
                        ):
        with open(logfile) as lf:
            if 'Total wall time' in lf.read():
                return True
            else:
                return False
            

    def gcbh_finished(
                      self,
                      logfile,
                      ):
        with open(logfile) as lf:
            if 'The best solution has not improved' in lf.read():
                return True
            else:
                return False


    def qe_finished(
                    self,
                    qe_output,
                    ):
        with open(qe_output) as qo:
            if ' JOB DONE.' in qo.read():
                return True
            else:
                return False
    

    def vasp_finished(
                    self,
                    vasp_output,
                    ):
        with open(vasp_output) as qo:
            if 'Total CPU time used' in qo.read():
                return True
            else:
                return False


    def energy_lammps(
                      self,
                      logfile,
                      ):
        initial_energy = 0
        with open(logfile) as lf:
            line = lf.readline()
            while line:
                if 'Per MPI rank' in line and initial_energy == 0:
                    line = lf.readline()
                    line = lf.readline()
                    initial_energy = float(line.split()[-2])

                elif ' Energy initial, next-to-last, final ' in line:
                    line = lf.readline()
                    final_energy = float(line.split()[-1])
                line = lf.readline()
        return initial_energy, final_energy


    def read_MD_trajectories(
                             self,
                             logfile,
                             ):
        trajectories = []
        times = [[0]]
        with open(logfile) as lf:
            line = lf.readline()
            while line:
                if line.startswith('dump'):
                    print(line)
                    dumpfile = line.split()[5]
                    dumpfile = os.path.join(os.path.dirname(logfile),dumpfile)
                    t = ase.io.read(dumpfile,format='lammps-dump-text',index=':')
                    for s in t:
                        s.pbc = True
                        s.set_atomic_numbers([self.lammps_binding[n] for n in s.get_atomic_numbers()])
#                    t = t[::20]
                    if len(t) == 501:
                        t = t[:-1]
                        print('Caution this is hardcoded, because I fogot to undump the general trajectory!')
                    trajectories.append(t)
                    timestemp = int(line.split()[4])
                    times.append(list((np.arange(len(t)) + 1) * timestemp + times[-1][-1]))
                line = lf.readline()
        times.pop(0)

        return trajectories, times


    # def parse_log(
    #               self,
    #               ):
    #     parsed_logs = {}
    #     for l,f in self.folders.items():
    #         logfile = os.path.join(f,self.md_files['logfile'])
    #         parsed_logs[l] = parse_lammps(logfile,printit=False)
    #     self.parsed_logs = parsed_logs


    def write_final(
                    self,
                    logfile,
                    output_name = 'final.in',
                    ):
        with open(logfile) as lf:
            line = lf.readline()
            while line:
                if line.startswith('dump'):
                    dumpfile = line.split()[5]
                line = lf.readline()
        dumpfile = os.path.join(os.path.dirname(logfile),dumpfile)
        final = ase.io.read(dumpfile,format='lammps-dump-text',index='-1')
        final.pbc = True
        final.set_atomic_numbers([self.lammps_binding[n] for n in final.get_atomic_numbers()])
        ase.io.write(output_name,final)


    def write_hot(
                  self,
                  output_name = 'hot.in',
                  ):
        logfile = self.md_files['logfile']
        for f in self.folders.values():
            os.chdir(f)
            temperatures = []
            snapshots = []
            collect = False
            with open(logfile) as lf:
                line = lf.readline()
                while line:
                    if line.startswith('minimize'):
                        collect = False
                    elif line.startswith('run'):
                        collect = True
                    elif line.startswith('dump') and not collect:
                        dumpfile = line.split()[5]
                        number_snapshots = int(line.split()[4])
                    elif line.startswith('Step') and collect:
                        tempid = np.argwhere(np.array(line.split())=='Temp')[0,0]
                        while line:
                            try:
                                line = lf.readline()
                                temperatures.append(float(line.split()[tempid]))
                                snapshots.append(int(line.split()[0]))
                            except:
                                break
                    line = lf.readline()
            hs = snapshots[np.argmax(temperatures)]/number_snapshots - 2
            dumpfile = os.path.join(os.path.dirname(logfile),dumpfile)
            hot = ase.io.read(dumpfile,format='lammps-dump-text',index=':')[hs]
            hot.pbc = True
            hot.set_atomic_numbers([self.lammps_binding[n] for n in hot.get_atomic_numbers()])
            ase.io.write(output_name,hot)

        os.chdir(self.homedir)


    def uncertainty_analysis(
                             self,
                             ):
        import generation_analysis as ga
        # vmin = self.kappa_min - 0.025
        # vmax = self.kappa_min + 0.025
        self.selected_folders = {}
        self.uncertainties = {}

        for label, folder in self.folders.items():
            print('{}'.format(label))
            os.chdir(folder)
            structures = glob.glob("*.xyz")
            for name in structures:
                print('{}'.format(name))
                # os.makedirs("{}".format(name.split(".")[0]), exist_ok=True)
                uncertain_structure = []
                uncertain_dissimilar_structure = []
                structure = ase.io.read(name, ":")
                for s in structure:
                    if max(s.arrays['local_sigma']) >= self.uncertainty_min:
                        uncertain_structure.append(s)
                if len(uncertain_structure) != 0:
                    uncertain_structure = ga.similarity.update_structure(uncertain_structure)
                    max_uncertain = max(uncertain_structure, key=lambda x: max(x.arrays['local_sigma']))
                    uncertain_dissimilar_structure.append(max_uncertain)
                    uncertain_structure.remove(max_uncertain)
                    # print(len(uncertain_dissimilar_structure), len(uncertain_structure))
                    while len(uncertain_structure) != 0:
                        sparsed_structure = []
                        similar_structures = []
                        for n1, s1 in enumerate(uncertain_structure):
                            score = ga.similarity.quick_score(s1,uncertain_dissimilar_structure[-1])
                            if np.min(score) <= self.kappa_min:
                                similar_structures.append(n1)
                            # scores = []
                            # for s2 in uncertain_dissimilar_structure:
                            #     scores.append(ga.similarity.quick_score(s1,s2))
                            # print("min max similarities:", min(scores), max(scores))
                            # for n2, score in enumerate(scores):
                            #     if np.min(score) <= self.kappa_min and n2 not in similar_structures:
                            #         similar_structures.append(n2)
                                # print(len(similar_structures))
                        # print(similar_structures)
                        sparsed_structure = [uncertain_structure[i] for i in range(len(uncertain_structure)) if i not in similar_structures]
                        # print(len(sparsed_structure))
                        if len(sparsed_structure) != 0:
                            max_uncertain = max(sparsed_structure, key=lambda x: max(x.arrays['local_sigma']))
                            uncertain_dissimilar_structure.append(max_uncertain)
                            sparsed_structure.remove(max_uncertain)
                        uncertain_structure = sparsed_structure
                        # print(len(uncertain_dissimilar_structure), len(uncertain_structure))
                print("Total {} number of structures are parsed".format(len(uncertain_dissimilar_structure)))
                if len(uncertain_dissimilar_structure) > 0:
                    self.uncertainties["{}-{}".format(label, name.split(".")[0])] = []
                    for a in uncertain_dissimilar_structure:
                        a.set_calculator()
                        self.uncertainties["{}-{}".format(label, name.split(".")[0])].append(max(a.arrays['local_sigma']))
                    os.makedirs("{}".format(name.split(".")[0]), exist_ok=True)
                    ase.io.write("{}/structures.xyz".format(name.split(".")[0]), uncertain_dissimilar_structure)
                    self.selected_folders["{}-{}".format(label, name.split(".")[0])] = os.path.join(self.folders[label], name.split(".")[0])
        os.chdir(self.homedir)


    def similarity_analysis(
                            self,
                            output_filename = 'similarity',
                            entire_training_set = False,
                            ):

        import generation_analysis as ga
        vmin = self.kappa_min - 0.025
        vmax = self.kappa_min + 0.025

        similarities = {}
        similarities_entire = {}
        if entire_training_set:
            slab_training = []
            for s in self.training_set:
                if s.info['structure_info'] != 'bulk' and s.info['structure_info'] != 'atom':
                    slab_training.append(s)

        for label, folder in self.folders.items():
            print('%s'%label)
            os.chdir(folder)
            structure = ase.io.read(self.md_files['final_structure'])
            structure = ga.similarity.update_structure(structure)

            specific_training = []
            for s in self.training_set:
                if s.info['structure_info'] == label:
                    specific_training.append(s)

            ase.io.write('training_%s.xyz'%label,specific_training)
            if len(specific_training) > 0:
                if ga.similarity.different_upper_lower(structure):
                    print('%s: Two different surfaces detected for test structure'%label)
                    split_surface = {}
                    split_surface_entire = {}
                    split_surface['upper'] = structure[structure.positions[:,2] > structure.cell[2,2]/2]
                    split_surface['lower'] = structure[structure.positions[:,2] < structure.cell[2,2]/2]
    
                    for side in ['upper','lower']:
                        s = split_surface[side]
                        similarity = ga.similarity.structure2training(
                                                                      s,
                                                                      specific_training,
                                                                      )
                        side_best_match = ga.similarity.best_match(
                                                                   similarity,
                                                                   specific_training,
                                                                   )
                        ase.io.write('%s_best_match.in'%side,side_best_match)
                        output = side + '_' + output_filename
                        ga.similarity.plot_single(
                                                  similarity,
                                                  vmin  = vmin,
                                                  vmax = vmax,
                                                  output_filename = side + '_' + output_filename,
                                                  )
                        split_surface[side] = similarity
                        if entire_training_set:
                            split_surface_entire[side] = ga.similarity.structure2training(
                                                                                          s,
                                                                                          slab_training,
                                                                                          )
                    similarities_entire[label] = split_surface_entire
                    similarities[label] = split_surface
                else:
                    similarity = ga.similarity.structure2training(
                                                                  structure,
                                                                  specific_training,
                                                                  )
                    best_match = ga.similarity.best_match(similarity,specific_training)
                    ase.io.write('best_match.in',best_match)
                    output = output_filename
                    similarities[label] = similarity
                    if entire_training_set:
                        similarity = ga.similarity.structure2training(
                                                                      structure,
                                                                      slab_training,
                                                                      )
                        similarities_entire[label] = similarity
                ga.similarity.plot_single(
                                          similarity,
                                          vmin  = vmin,
                                          vmax = vmax,
                                          output_filename = output,
                                          )

        self.similarities = similarities
        self.similarities_entire_training = similarities_entire
        os.chdir(self.homedir)


    def write_final_MD_structures(
                                   self,
                                   ):
        final_structures = []
        for label, folder in self.folders.items():
            s = 'final.in'
            atom = ase.io.read(os.path.join(folder,s))
            atom.info['structure_info'] = label
            final_structures.append(atom)
        ase.io.write('final_MD_structures.xyz',final_structures)


    def write_selected_for_DFT(
                               self,
                               ):
        selected_structures = []
        if len(self.selected_folders) > 0:
            for label, folder in self.selected_folders.items():
                s = 'structures.xyz'
                atom = ase.io.read(os.path.join(folder,s), ":")
                for a in atom:
                    a.info['structure_info'] = label
                    a.set_calculator()
                    del a.arrays['SOAP']
                    selected_structures.append(a)
            self.selected_MD_structures = selected_structures
            ase.io.write('selected_MD_structures.xyz',self.selected_MD_structures)
        # else:
        #     for label, folder in self.folders.items():
        #         s = 'final.in'
        #         atom = ase.io.read(os.path.join(folder,s))
        #         atom.info['structure_info'] = label
        #         selected_structures.append(atom)
        #     self.selected_MD_structures = selected_structures
        #     ase.io.write('selected_MD_structures.xyz',self.selected_MD_structures)


    def generate_all_DFT_data(
                              self,
                              ):
        if len(self.selected_folders) > 0:
            for label_folder in self.selected_folders.items():
                print(label_folder)
                structures = 'structures.xyz'
                self.generate_DFT_data(
                                       label_folder,
                                       structures,
                                       )
        else:
            for label_folder in self.folders.items():
                print(label_folder)
                structures = 'structures.xyz'
                self.generate_DFT_data(
                                       label_folder,
                                       structures,
                                       )
        self.write_dft_starter()


    def write_dft_starter(self):
        file = open(os.path.join(self.dftdir,'start_dft.sh'),'w')
        file.write("""curdir=$(pwd)
controlfile=$curdir/dft.cmd
directories=$(find . -type f -name "POSCAR" -exec dirname {} \;)
for d in $directories; do
        cd $d
        m=$(echo "$d" | tr -d ./_psrdanonial)
        jobname=$m
        echo $m
        cp $controlfile control.cmd
        sed -i "s/JOBNAME/$jobname/" control.cmd
        sed -i "s/WALLTIME/06/" control.cmd
        qsub control.cmd
        cd $curdir
done
                   """)
        file = open(os.path.join(self.dftdir,'dft.cmd'),'w')
        file.write("""#!/bin/sh
#PBS -l select=25
#PBS -l place=scatter
#PBS -l walltime=WALLTIME:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q prod
#PBS -N JOBNAME
#PBS -A CatDynEnsemble

source ~/.bashrc
conda activate tio2
module purge
module load nvhpc/23.3
module load PrgEnv-nvhpc/8.3.3
module load cray-libsci/23.02.1.1
module load craype-accel-nvidia80

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/23.3/compilers/extras/qd/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/libraries/aocl/3.2.0/lib

# Change to working directory
cd ${PBS_O_WORKDIR}

export MPICH_GPU_SUPPORT_ENABLED=1
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=2
NDEPTH=4
NTHREADS=4
NGPUS=2
NTOTRANKS=$(( NNODES * NRANKS ))

export ASE_VASP_COMMAND="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth ${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} /home/ylee/codes/VASP/source/vasp.6.4.1/bin/vasp_std"
export VASP_PP_PATH=/home/ylee/codes/VASP/source

python run_vasp.py                   
                   """)

    def generate_DFT_data(
                          self,
                          label_folder,
                          structures = 'structures.xyz',
                          ):
        label, folder = label_folder
        ls = label.split('-')
        print(ls)
        atom = ase.io.read(os.path.join(folder,structures), ":")
        for n, a in enumerate(atom):
            nn = f'{n:03d}'
            if len(ls) == 3:
                tpath = os.path.join(self.dftdir,ls[0],ls[1],ls[2],nn)
            else:
                tpath = os.path.join(self.dftdir,ls[0],ls[1],nn)
            try:
                pathlib.Path(tpath).mkdir(parents=True)
            except OSError:
                pass

            ase.io.vasp.write_vasp(os.path.join(tpath,self.dft_files['input_structure']), a, vasp5=True, sort=True)

            file = open(os.path.join(tpath,self.dft_files['run_script']),'w')
            file.write("""import ase, ase.io
import ase.io.vasp
from ase.calculators.vasp import Vasp
import scipy.linalg as LA
import numpy as np
import sys

def kpoint_grid(cell, dx, fd_info=sys.stdout):
    rc = cell.cell.reciprocal()
    l_rc = np.empty(3,dtype=np.float64)
    for i in range(3):
        l_rc[i] = LA.norm(rc[i], 2)
    v = l_rc[:]/dx

    k_grid_large = list(map(int, np.ceil(v)))
    dx_large = l_rc[:]/list(map(float, k_grid_large[:]))
    k_grid_small =  list(map(int, np.floor(v)))
    k_grid_rounded =  list(map(int, np.rint(v)))
    for i in range(3):
        if k_grid_small[i] == 0:
            k_grid_small[i] = 1
        if k_grid_rounded[i] == 0:
            k_grid_rounded[i] = 1
    dx_small = l_rc[:]/list(map(float, k_grid_small[:]))
    dx_rounded = l_rc[:]/list(map(float, k_grid_rounded[:]))
    print(k_grid_rounded)
    return k_grid_rounded
                       
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
     sigma=    0.1,
     nelm=     500,
     nsw=      0,
     ispin=    2,
     ibrion=   -1,
     lwave=    False,
     lcharg=   False,
     lreal=    False,
     lasph=    True,
     lorbit=   10,
     ivdw=     12,
     setups=   {'base': 'recommended'},
     algo=     'All',
     ldau_luj= {'Zr': {'L': 2, 'U': 4, 'J': 0},
                'O':  {'L': -1, 'U': 0, 'J': 0},
                'H':  {'L': -1, 'U': 0, 'J': 0},
                'Cu':  {'L': -1, 'U': 0, 'J': 0},
                'Pd':  {'L': -1, 'U': 0, 'J': 0},
                },
     lmaxmix=  4,
     ldauprint=0,
     ldautype= 2,
     ncore=    32,
     )

atom = ase.io.read("POSCAR")
kpts = kpoint_grid(atom, 0.04)
calc.set(kpts=kpts)
atom.calc = calc

atom.get_potential_energy()
                   
                       """)
            
        os.chdir(self.homedir)


    def get_rid_of_K_in_z(
                          self,
                          qe_input,
                          ):
        tfile = os.path.join(self.homedir,'temp.inp')
        fout = open(tfile,'wt')
        fin = open(qe_input,'r')
        line = fin.readline()
        while line:
            if 'K_POINTS' in line:
                fout.write(line)
                line = fin.readline()
                sw = line.split()
                fout.write('%s %s 1 0 0 0\n'%(sw[0],sw[1]))
                line = fin.readline()
            fout.write(line)
            line = fin.readline()
        fout.close()
        os.system('mv %s %s'%(tfile,qe_input))


    def extract_forces(
                       self,
                       ):
        self.dft_scores = {}
        keeper = []
        for label, folder in self.dft_folders.items():
            print('\nChecking DFT results of %s'%label)
            for key in folder:
                keeper += self.check_dft(
                                         folder[key],
                                         label = label,
                                         )
                print("{}: {} is successfully processed!".format(label, key))
        self.add_forces = keeper
        add_forces = []
        for s in self.add_forces:
            for x in ['SOAP','structural_info']:
                try:
                    del s.arrays[x]
                except:
                    pass
            add_forces.append(s)
        for atom in add_forces:
            z_slab_min = np.min(atom.positions[:,2])
            force_mask = []
            for a in atom:
                if a.position[2] < z_slab_min + 5:
                    force_mask.append(True)
                else:
                    force_mask.append(False)
            atom.arrays['force_mask'] = np.array(force_mask)
        ase.io.write('add_forces.xyz',add_forces)


    def check_dft(
                  self,
                  folder,
                  label,
                  ):
        if not all([os.path.isfile(os.path.join(folder,f)) for f in self.dft_files.values()]):
            print(colored('DFT files not found', 'green'))
            return []
        if not self.vasp_finished(os.path.join(folder,self.dft_files['output'])):
            print(colored('DFT not yet done', 'green'))
            return []
        s = ase.io.read(os.path.join(folder,self.dft_files['input_structure']))
        result = process_ase(
                             s,
                             os.path.join(folder,'OUTCAR'),
                             qeq = False,
                             )
        s = result[0]
        for k in s:
            k.info['structure'] = 'slab'
            k.info['structure_info'] = label
        ase.io.write(os.path.join(folder,'forces.xyz'),s)
        ase.io.write(os.path.join(folder,'final.in'),s[-1])
        return s


    def extract_lowest_energy_structures(
                                         self,
                                         ):
        min_structures = {}
        for s in self.training_set:
            info = s.info['structure_info']
            if info != 'bulk' and info != 'atom' and not 'peroxo' in info:
                if info not in min_structures:
                    min_structures[info] = s
                else:
                    new_energy = s.info['dft_energy']
                    old_energy = min_structures[info].info['dft_energy']
                    if new_energy < old_energy:
                        min_structures[info] = s
        self.min_structures = min_structures
        compdir = os.path.join(self.homedir,'sfe_structures')
        try:
            os.mkdir(compdir)
        except OSError:
            pass
        for l,s in self.min_structures.items():
            ase.io.write(os.path.join(compdir,'%s.in'%l),s)
            for x in ['SOAP','structural_info']:
                try:
                    del s.arrays[x]
                except:
                    pass
            ase.io.write(os.path.join(compdir,'%s.xyz'%l),s)


    def get_atomic_energies(
                            self,
                            ):
        atomic_energies = {}
        for s in self.training_set:
            if len(s) == 1:
                atomic_energies[s.get_chemical_formula()] = s.info['dft_energy']
        self.atomic_energies = atomic_energies


    def get_relative_DFT_energies(
                                  self,
                                  reference='',
                                  labels = [],
                                  ):
        if len(labels) == 0:
            labels = self.folders.keys()
        energies = {}
        numbers = {}
        results = pd.DataFrame()
        for s in self.training_set:
            l = s.info['structure_info']
            if not l in labels:
                continue
            e = s.info['dft_energy']
            for el,ae in self.atomic_energies.items():
                 noa = np.sum(np.array(s.get_chemical_symbols()) == el)
                 e -= noa*ae
            e /= len(s)
            if l not in numbers:
                numbers[l] = 1
            else:
                numbers[l] += 1
            if numbers[l] == 1:
                energies[l] = [e,0,e]
            if numbers[l] == 2:
                energies[l][1] = e
                energies[l][2] = e
            else:
                if e < energies[l][2]:
                    energies[l][2] = e
        results =  pd.DataFrame(energies).transpose()
        results.reset_index(inplace=True)
        results.rename(columns={'index':'label',0:'dft_initial',1:'dft_final',2:'dft_complexion'},inplace=True)
        results.loc[(results['dft_complexion'] >= results['dft_final']),'dft_complexion'] = np.NaN
        results.to_csv('relative_energies.csv')
        try:
            ref_res = pd.read_csv(reference)
            ref_res.rename(columns={'dft_complexion':'reference'},inplace=True)
            results = results.merge(ref_res[['label','reference']],on='label')
        except IOError:
            pass
        if hasattr(self, 'relative_energies'):
            self.relative_energies = self.relative_energies.merge(results,on='label')
        else:
            self.relative_energies = results
        # plot_relative_DFT_energies(results)


# def plot_scores_vs_old(scores):
#     plt.plot(scores.values(),color='black',marker='x',linestyle='')
#     plt.xticks(np.arange(len(scores)),scores.keys(),rotation='vertical',)
#     plt.subplots_adjust(bottom=0.20,left=0.15)
#     plt.xlim(-1,len(scores))
#     plt.ylim(0)
#     plt.ylabel(r'$\kappa$(new,old)')
#     plt.savefig('similarity2old_complexions')
#     plt.close()


# def plot_relative_DFT_energies(
#                                results,
#                                ):

#     results['dft_final'].plot(color='blue',label='relaxed',marker='o',linestyle='')
#     results['dft_initial'].plot(color='red',label='bulk-truncated',marker='x',linestyle='')
#     if 'reference' in results:
#         results['reference'].plot(color='black',label='reference',marker='d',linestyle='')
#     results['dft_complexion'].plot(color='pink',label='complexion',marker='*',linestyle='')
#     plt.legend()
#     plt.xticks(
#               np.arange(results.shape[0]),
#               results['label'],
#               rotation='vertical',
#               )
#     plt.subplots_adjust(bottom=0.30,left=0.15)
#     plt.xlim(-1,len(results))
#     plt.savefig('relative_energies',dpi=300)
#     plt.close()


# def plot_GAPvsDFT_energies(
#                            results,
#                            output_name='GAPvsDFT_formation_energies',
#                            sfe=False,
#                            ):
#     colors={'initial':'red','final':'blue'}
#     labels={'initial':'bulk-truncated','final':'relaxed'}
#     for snapshot in ['initial','final']:
#         x = results['dft_%s'%snapshot]
#         y = results['GAP_%s'%snapshot]
#         if sfe:
#             x /= results['surface_area']
#             y /= results['surface_area']

#         plt.plot(
#                  results['dft_%s'%snapshot],
#                  results['GAP_%s'%snapshot],
#                  marker = 'o',
#                  linestyle = '',
#                  color = colors[snapshot],
#                  mfc='none',
#                  label = labels[snapshot],
#                  markersize=10,
#                  markeredgewidth=2,
#                 )
#     plt.ylabel(
#                r'$ E^{\mathrm{GAP}}_{\mathrm{coh}} (\mathrm{eV}/\mathrm{atom})$',
#                fontsize='16',
#                )
#     ax = plt.gca()
#     plt.xlabel(
#                r'$ E^{\mathrm{DFT}}_{\mathrm{coh}} (\mathrm{eV}/\mathrm{atom})$',
#                fontsize='16',
#                )
#     ylim = plt.ylim()
#     xlim = plt.xlim()
#     ll = np.min([xlim[0],ylim[0]])
#     hl = np.max([xlim[1],ylim[1]])
#     plt.xlim(ll,hl)
#     plt.ylim(ll,hl)
#     plt.plot([ll,hl],[ll, hl],color='black',linestyle='--')
#     plt.legend(fontsize='12')
#     plt.subplots_adjust(left=0.22,bottom=0.15)
#     plt.savefig(output_name,dpi=300)#,bbox_inches='tight')
#     #plt.savefig(output_name + '.pdf',dpi=1000)#,bbox_inches='tight')
#     plt.close()


# def plot_energy_difference(
#                            results,
#                            output_name='GAP_DFT_energy',
#                            sfe=False,
#                            ):
#     colors={'initial':'red','final':'blue'}
#     labels={'initial':'bulk-truncated','final':'relaxed'}
#     results = update_labels(results)

#     fig, ax1 = plt.subplots()

#     lns = []
#     for snapshot in ['initial','final']:
#         x = results['dft_%s'%snapshot]
#         y = results['GAP_%s'%snapshot]
#         if sfe:
#             x /= results['surface_area']
#             y /= results['surface_area']
#         results['diff_%s'%snapshot] = x - y
#         tlns = ax1.plot(
#                         results['diff_%s'%snapshot].abs()*1000,
#                         marker = 'x',
#                         linestyle = '',
#                         color = colors[snapshot],
#                         mfc='none',
#                         label = labels[snapshot],
#                         markersize=10,
#                         markeredgewidth=2,
#                        )
#         lns.append(tlns)
#     plt.legend(fontsize='12',loc='upper left')
#     ax1.set_ylabel(
#                    r'$|\gamma_{\mathrm{surf}}^{(hkl),\sigma}|$ $(\mathrm{meV}/\mathrm{\AA}^2)$',
#                    #r'$\Delta E_{\mathrm{coh}} (\mathrm{meV}/\mathrm{atom})$',
#                    fontsize='16',
#                    )
#     plt.xticks(
#                np.arange(results.shape[0]),
#                results['label-fancy'],
#                rotation='vertical',
#                 )
#     ax2 = ax1.twinx()
#     ax2.set_ylabel(r'$\kappa_{\mathrm{DFT,GAP}}$',fontsize='16')
#     tlns = ax2.plot(
#                     results['similarity_score'],
#                     color = 'green',
#                     label = r'$\kappa_{\mathrm{DFT,GAP}}$',
#                     linestyle = '',
#                     marker = 'd',
#                     )
#     lns = lns[0] + lns[1] + tlns
#     labs = [l.get_label() for l in lns]
#     ax2.set_ylim(0,0.2)
# #    ax1.set_ylim(0,115)

#     ax1.legend(lns, labs, loc=0,fontsize='12')
#     plt.subplots_adjust(left=0.12,bottom=0.30,right=0.85)
#     plt.savefig('difference_' + output_name,dpi=300)#,bbox_inches='tight')
#     #plt.savefig(output_name + '.pdf',dpi=1000)#,bbox_inches='tight')
#     plt.close()




# def plot_energy_similarity(
#                            similarities,
#                            energies,
#                            training_energy = 0,
#                            output_name = 'geoopt_similiarity_energy',
#                            soaplim = 0.25
#                            ):

#     fig, ax1 = plt.subplots()
#     ax1.set_xlabel(r'Snapshot $i$')
#     ax1.set_ylabel(r'$E_{\mathrm{rel}}$ ($\gamma_{\mathrm{surf}}^{(hkl),\sigma}) \ (\mathrm{meV}/\mathrm{\AA}^2)$')
#     ax1.plot(
#              energies,
#              color = 'black',
#              label = r'$E_{\mathrm{rel}}$',
#              )
#     if training_energy:
#         plt.axhline(
#                     training_energy,
#                     linestyle=':',
#                     color = 'grey',
#                     linewidth = 1.5,
#                     )
#     ax2 = ax1.twinx()
#     ax2.set_ylabel(r'$\kappa(i,C_k^{(hkl)-\sigma}$')
#     ax2.plot(
#              similarities,
#              color = 'green',
#              label = r'$\kappa_{\mathrm{i,GAP_{min}}}$',
#              linestyle = '--',
#              )
#     ax2.set_ylim(0,soaplim)
#     ax1.set_xlim(0,len(energies))
#     plt.subplots_adjust(right=0.85)
#     plt.subplots_adjust(left=0.15)
#     plt.savefig(output_name,dpi=300)
#     plt.close()


# def plot_scores(
#                 scores,
#                 nmax = 20,
#                 output_name = 'geoopt_score',
#                 style = 'training',
#                 soaplim = 0.25,
#                 ):
#     colors = {
#               '001' : 'black',
#               '010' : 'red',
#               '011' : 'blue',
#               '110' : 'yellow',
#               '111' : 'green',
#               }
#     linestyles = {
#                   't0' : ':',
#                   't1' : '-',
#                   't2' : '--',
#                   't3' : '--',
#                   }

#     for label,score in scores.items():
#         try:
#             score[0]
#         except:
#             score = score[style]
#         ls = label.split('-')
#         color = colors[ls[0]]
#         lsty = linestyles[ls[1]]
#         if label == '111-t2':
#             lsty = '-'
#         plt.plot(
#                  score,
#                  color=color,
#                  label = label,
#                  linestyle=lsty,
#                  )
#     plt.xlim(0,nmax)
#     plt.ylim(0,soaplim)
#     plt.legend( bbox_to_anchor=(0.6,0.5))
#     plt.ylabel(r'$\kappa(B_0^{(hkl)-\sigma},C_1^{(hkl)-\sigma}$')
#     plt.xlabel(r'Snapshot $i$')
#     plt.xticks(np.arange(0,nmax+1,5))
#     plt.tight_layout()
#     plt.savefig(output_name,dpi=300)
#     plt.close()


# def update_labels(
#                   sfedata,
#                   labeldic = {
#                               '001-t0': '(001) M-rich',
#                               '001-t1': '(001) stoich',
#                               '001-t2': '(001) O-rich',
#                               '010-t0': '(010) M-rich',
#                               '010-t1': '(010) stoich',
#                               '010-t2': '(010) O-rich',
#                               '011-t0': '(101) M-rich',
#                               '011-t1': '(101) stoich',
#                               '011-t2': '(101) O-rich',
#                               '110-t0': '(110) M-rich',
#                               '110-t1': '(110) stoich',
#                               '110-t2': '(110) O-rich',
#                               '111-t0': '(111) M-rich',
#                               '111-t1': '(111) stoich$_1$',
#                               '111-t2': '(111) stoich$_2$',
#                               '111-t3': '(111) O-rich',
#                             }
#                      ):
#     x = pd.DataFrame(columns=['label','label-fancy'])
#     x['label'] = labeldic.keys()
#     x['label-fancy'] = labeldic.values()
#     return sfedata.merge(x)

