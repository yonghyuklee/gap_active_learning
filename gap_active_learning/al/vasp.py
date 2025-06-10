import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import ase, ase.io, ase.io.vasp
import pandas as pd
import pathlib
import glob
from termcolor import colored

from gap_active_learning.parser.vasp2ase import process_ase
from gap_active_learning.parser.cluster import *

import gap_active_learning.al as ga
from gap_active_learning.setups.vasp import *


# MLP: GAP Class
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
                 project = 'CuPd',

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
        self.project = project

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
                        labels.append(l)

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
                if os.path.isfile(logfile):
                    if self.gcbh_finished(logfile):
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        labels.append(l)
                    else:
                        print("GCBH is not converged!!")
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        labels.append(l)

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
                    if len(t) == 501:
                        t = t[:-1]
                        print('Caution this is hardcoded, because I fogot to undump the general trajectory!')
                    trajectories.append(t)
                    timestemp = int(line.split()[4])
                    times.append(list((np.arange(len(t)) + 1) * timestemp + times[-1][-1]))
                line = lf.readline()
        times.pop(0)

        return trajectories, times


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
        self.selected_folders = {}
        self.uncertainties = {}

        for label, folder in self.folders.items():
            print('{}'.format(label))
            os.chdir(folder)
            structures = glob.glob("*.xyz")
            for name in structures:
                print('{}'.format(name))
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
                    while len(uncertain_structure) != 0:
                        sparsed_structure = []
                        similar_structures = []
                        for n1, s1 in enumerate(uncertain_structure):
                            score = ga.similarity.quick_score(s1,uncertain_dissimilar_structure[-1])
                            if np.min(score) <= self.kappa_min:
                                similar_structures.append(n1)
                        sparsed_structure = [uncertain_structure[i] for i in range(len(uncertain_structure)) if i not in similar_structures]
                        if len(sparsed_structure) != 0:
                            max_uncertain = max(sparsed_structure, key=lambda x: max(x.arrays['local_sigma']))
                            uncertain_dissimilar_structure.append(max_uncertain)
                            sparsed_structure.remove(max_uncertain)
                        uncertain_structure = sparsed_structure
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


    def generate_DFT_data(
                          self,
                          label_folder,
                          structures={'final': 'final.xyz'},
                         ):
        label, folder = label_folder
        ls = label.split('-')
        print(ls)
    
        for n, s in structures.items():
            atoms = ase.io.read(os.path.join(folder, s), ':')
            for i, atom in enumerate(atoms):
                # Generalized path generation
                tpath = os.path.join(self.dftdir, *ls, n, "%.3d" % i)
    
                # Ensure the directory exists
                pathlib.Path(tpath).mkdir(parents=True, exist_ok=True)
    
                # Write structure file
                ase.io.vasp.write_vasp(
                    os.path.join(tpath, self.dft_files['input_structure']),
                    atom,
                    vasp5=True,
                    sort=True,
                )
    
                # Write run script
                with open(os.path.join(tpath, self.dft_files['run_script']), 'w') as file:
                    file.write(generate_vasp_script(project=self.project))
    
        os.chdir(self.homedir)


    def extract_forces(self):
        self.dft_scores = {}
        keeper = []
        for label, folder in self.dft_folders.items():
            print(f'\nChecking DFT results of {label}')
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
                             os.path.join(folder,self.dft_files['output']),
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


# MLP: MACE Class
class MACEGen:

    def __init__(
                 self,
                 dft_files = {
                             'input_structure' :'POSCAR',
                             'output' :'OUTCAR',
                             'run_script': 'run_vasp.py',
                             },
                 md_files = {
                             'final_structure' :'md.xyz',
                             },
                 md_dir = 'md/',
                 max_selected = 100,
                 nn_uncertainty = 0.1,
                 max_force = 10,
                 project = 'CuPd',
                 foundation = False,
                 ):

        self.homedir = os.path.abspath('.')
        self.max_selected = max_selected
        self.mddir = os.path.join(self.homedir,md_dir)
        self.nn_uncertainty = nn_uncertainty
        self.max_force = max_force
        self.foundation = foundation

        if self.mddir[-1] != '/':
            self.mddir+= '/'
        self.dftdir = os.path.join(self.homedir,'dft/')
        self.dft_files = dft_files
        self.md_files = md_files
        self.project = project

        self.get_md_folders()


    def get_md_folders(
                       self,
                       ):
        self.folders = {}
        self.md_energies = pd.DataFrame()
        final_energies = []
        labels = []

        for dinfo in os.walk(self.mddir):
            d = dinfo[0]
            output_name = os.path.join(d,self.md_files['final_structure'])
            if os.path.isfile(output_name):
                try:
                    final = ase.io.read(output_name)
                    l = d.replace(self.mddir,'').replace('/','-')
                    self.folders[l] = os.path.join(self.homedir,d)
                    fe = final.get_potential_energy()
                    final_energies.append(fe)
                    labels.append(l)
                except:
                    pass
        self.md_energies['label'] = labels
        self.md_energies['final'] = final_energies


    def get_dft_folders(self):
        self.dft_folders = {}
        for root, _, _ in os.walk(self.dftdir):
            dft_output = os.path.join(root, self.dft_files['input_structure'])
            if os.path.isfile(dft_output):
                path = pathlib.Path(root).parent.resolve()
                folder_extension = pathlib.Path(root).name
                label = str(path.relative_to(self.dftdir)).replace('/', '-')

                # Use setdefault to simplify appending subfolders
                self.dft_folders.setdefault(label, {})[folder_extension] = str(root)


    def vasp_finished(self, vasp_output):
        try:
            with open(vasp_output) as qo:
                content = qo.read()
            if 'aborting loop EDIFF was not reached (unconverged)' in content:
                print(colored('DFT not converged in NELM', 'red'))
                return False
            if 'Total CPU time used' in content:
                # print(colored('DFT FINISHED', 'green'))
                return True
            print(colored('DFT status unknown', 'yellow'))
            return False
        except FileNotFoundError:
            print(colored(f'File not found: {vasp_output}', 'red'))
            return False


    def generate_DFT_data_from_uncertainty(
                                           self,
                                           cluster=False,
                                           n_cluster=10,
                                          ):
        selected_structures = []
        for label_folder in self.folders.items():
            s = self.md_files['final_structure']
            label, folder = label_folder
            print(f"MLP simulated structures read from {folder}")
            atoms = ase.io.read(os.path.join(folder,s), ":")

            FE_present = all('FE' in atom.info for atom in atoms)

            if not self.foundation:
                std_devs = []
                for i, a in enumerate(atoms):
                    try:
                        a_f = a.arrays['MACE_fitA_forces']
                        b_f = a.arrays['MACE_fitB_forces']
                        c_f = a.arrays['MACE_fitC_forces']
                        sigmas = []
                        for fa, fb, fc in zip(a_f, b_f, c_f):
                            av = ( fa + fb + fc ) / 3
                            component_wise_variances = np.mean(( ( fa - av ) ** 2 + ( fb - av ) ** 2 + (fc - av ) ** 2 ) / 3)
                            sigma = np.sqrt(component_wise_variances)
                            sigmas.append(sigma)
                        sigma = np.max(sigmas)
                        max_force = np.max(np.abs(a.arrays['MACE_forces'].flatten()))
                        std_devs.append({'ID':i, 'std':sigma, 'max_force':max_force})
                        if FE_present:
                            std_devs[-1]['FE'] = a.info['FE'] 
                        
                        a.info['max_sigma'] = sigma
                    except:
                        sigmas = a.arrays['std_dev']
                        sigma = np.max(sigmas)
                        max_force = np.max(a.arrays['MACE_forces'].flatten())
                        std_devs.append({'ID':i, 'std':sigma, 'max_force':max_force})
                        a.info['max_sigma'] = sigma
            
                df = pd.DataFrame(std_devs)
                df_order = df.sort_values(by=['std'], ascending=False)
                im_top = []
                for index, row in df_order.iterrows():
                    if len(im_top) == self.max_selected:
                        break
                    elif (row['std'] >= self.nn_uncertainty 
                          and row['max_force'] <= self.max_force 
                          and ga.similarity.examine_unconnected_components(atoms[int(row['ID'])])):
                        print(f"sigma: {row['std']}, maximum_force: {row['max_force']}", f", FE: {row['FE']}" if 'FE' in row else "")
                        im_top.append(atoms[int(row['ID'])])

            elif self.foundation:
                im_top = atoms

            if cluster and im_top:
                im_top = kpca_kmeans(im_top, kmeans_clusters=n_cluster, kernel='poly')
                
            if im_top:
                ase.io.write(os.path.join(folder,"final.xyz"), im_top)

                for atom in im_top:
                    atom.info['structure_info'] = label
                    selected_structures.append(atom)
                structures = {'final':'final.xyz'}
                self.generate_DFT_data(
                                       label_folder,
                                       structures,
                                       )
        self.selected_MD_structures = selected_structures
        ase.io.write('selected_MD_structures.xyz',self.selected_MD_structures)


    def generate_DFT_data(
                          self,
                          label_folder,
                          structures={'final': 'final.xyz'},
                         ):
        label, folder = label_folder
        ls = label.split('-')
        print(ls)
    
        for n, s in structures.items():
            atoms = ase.io.read(os.path.join(folder, s), ':')
            for i, atom in enumerate(atoms):
                # Generalized path generation
                tpath = os.path.join(self.dftdir, *ls, n, "%.3d" % i)
    
                # Ensure the directory exists
                pathlib.Path(tpath).mkdir(parents=True, exist_ok=True)
    
                # Write structure file
                ase.io.vasp.write_vasp(
                    os.path.join(tpath, self.dft_files['input_structure']),
                    atom,
                    vasp5=True,
                    sort=True,
                )
    
                # Write run script
                with open(os.path.join(tpath, self.dft_files['run_script']), 'w') as file:
                    file.write(generate_vasp_script(project=self.project))
    
        os.chdir(self.homedir)


    def extract_forces(self):
        self.dft_scores = {}
        keeper = []
        for label, folder in self.dft_folders.items():
            print(f'\nChecking DFT results of {label}')
            for key, subfolder in folder.items():
                keeper += self.check_dft(subfolder, label)
                # print(f"{label}: {key} successfully processed!")
        
        self.add_forces = keeper
        add_forces = []
        for s in self.add_forces:
            for key in ['SOAP', 'structural_info']:
                s.arrays.pop(key, None)  # Safely removes the key if it exists
            add_forces.append(s)
        
        ase.io.write('add_forces.xyz', add_forces)


    def check_dft(self, folder, label):
        required_files = [os.path.join(folder, f) for f in self.dft_files.values()]
        if not all(os.path.isfile(f) for f in required_files):
            print(colored(f'DFT files missing in {folder}', 'red'))
            return []

        if not self.vasp_finished(os.path.join(folder, self.dft_files['output'])):
            print(colored(f'DFT not yet completed for {folder}', 'red'))
            return []
    
        try:
            s = ase.io.read(os.path.join(folder, self.dft_files['input_structure']))
            result = process_ase(
                s,
                os.path.join(folder, 'OUTCAR'),
            )
            s = result[0]
        except Exception as e:
            print(colored(f'Error processing DFT results in {folder}: {e}', 'red'))
            return []
    
        for k in s:
            k.info['structure'] = 'slab'
            k.info['structure_info'] = label
    
        ase.io.write(os.path.join(folder, 'forces.xyz'), s)
        ase.io.write(os.path.join(folder, 'final.in'), s[-1])

        max_force = np.max(np.abs(s[-1].arrays['dft_forces'].flatten()))
        if max_force <= self.max_force:
            return [s[-1]]
        else:
            print(colored(f'Maximum force larger than the criteria in {folder}', 'red'))
            return []