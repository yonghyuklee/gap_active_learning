import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os, sys, argparse
import ase, ase.io
import re
import pandas as pd
import tempfile
import pickle
import pathlib
import ase.io.espresso
import operator
from termcolor import colored

import gap_active_learning.setups.qe as QEgenerator
from gap_active_learning.parser.qe2ase import process_ase

import gap_active_learning.al as ga


class GapGen:

    def __init__(
                 self,
                 gapfile,
                 trainingfile,
                 labels = [],
                 dft_files = {
                             'input_structure' :'structure.in',
                             'output' :'out',
                             },
                 md_files = {
                             'logfile' : 'log.lammps',
                             'final_structure' :'final.in',
                             },
                 md_dir = 'md/',

                 # NEED TO BE TESTED FOR YOUR SYSTEM
                 kappa_min = 0.075,
                 geoopt_maxsteps = 20,
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
        self.geoopt_maxsteps = geoopt_maxsteps
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

        self.get_md_folders(
                            labels,
                            )


    def get_md_folders(
                       self,
                       labels,
                       ):
        self.folders = {}
        self.md_energies = pd.DataFrame()
        initial_energies = []
        final_energies = []
        labels = []
        if len(labels) == 0:
            for dinfo in os.walk(self.mddir):
                d = dinfo[0]
                logfile = os.path.join(d,self.md_files['logfile'])
                output_name = os.path.join(d,self.md_files['final_structure'])
                if os.path.isfile(logfile):
                    if self.lammps_finished(logfile):
                        self.write_final(logfile,output_name=output_name)
                        l = d.replace(self.mddir,'').replace('/','-')
                        self.folders[l] = os.path.join(self.homedir,d)
                        ie,fe = self.energy_lammps(logfile)
                        final_energies.append(fe)
                        initial_energies.append(ie)
                        labels.append(l)
            self.md_energies['label'] = labels
            self.md_energies['initial'] = initial_energies
            self.md_energies['final'] = final_energies

        elif len(labels) > 1:
            for n,l in enumerate(labels):
                d = l.replace('-','/')
                d = os.path.join(self.mddir,d)
                if os.path.exists(d):
                    take = True
                    for k,f in self.md_files.items():
                        if not os.path.isfile(os.path.join(d,f)):
                            take = False
                            break
                        elif k == 'logfile':
                            if not self.lammps_finished(os.path.join(d,f)):
                                take = False
                                break
                    if take:
                        self.folders[l] = os.path.join(self.homedir,d)
                    else:
                        print('Requisites not available for %s'%d)


    def get_dft_folders(
                        self,
                        ):
        self.dft_folders = {}
        for dinfo in os.walk(self.dftdir):
            d = dinfo[0]
            dft_output = os.path.join(d,self.dft_files['output'])
            if os.path.isfile(dft_output):
                if self.qe_finished(dft_output):
                    path = os.path.abspath(d+'/../')
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


    def qe_finished(
                    self,
                    qe_output,
                    ):
        with open(qe_output) as qo:
            if ' JOB DONE.' in qo.read():
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


    def similarity_analysis(
                            self,
                            output_filename = 'similarity',
                            entire_training_set = False,
                            ):

        import gap_active_learning.al as ga
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


    def determine_selected_slabs(
                                 self,
                                 compare2entiretraining = False,
                                 ):
        take_away = {}
        for label,tsim in self.similarities.items():
            score = ga.similarity.calculate_score(tsim)
            if score > self.kappa_min:
                take_away[label] = score
            if compare2entiretraining == True:
                try:
                    etsim = self.similarities_entire_training[label]
                except KeyError:
                    print('%s not found for similarities vs entire training'%label)
                    continue
                escore = ga.similarity.calculate_score(etsim)
                if escore != score:
                    print('Different scores for subset and entire training set for %s'%label)
                    print('Entire Training Set: %s\nSubset: %s'%(np.round(escore,3),np.round(score,3)))
        take_away = dict(sorted(take_away.items(),key=operator.itemgetter(1),reverse=True)[:self.max_selected])
        self.selected_folders = {k:self.folders[k] for k in take_away.keys() if k in self.folders}


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
                s = 'final.in'
                atom = ase.io.read(os.path.join(folder,s))
                atom.info['structure_info'] = label
                selected_structures.append(atom)
            self.selected_MD_structures = selected_structures
            ase.io.write('selected_MD_structures.xyz',self.selected_MD_structures)
        else:
            for label, folder in self.folders.items():
                s = 'final.in'
                atom = ase.io.read(os.path.join(folder,s))
                atom.info['structure_info'] = label
                selected_structures.append(atom)
            self.selected_MD_structures = selected_structures
            ase.io.write('selected_MD_structures.xyz',self.selected_MD_structures)


    def generate_all_DFT_data(
                              self,
                              HT = True,
                              ):
        if len(self.selected_folders) > 0:
            for label_folder in self.selected_folders.items():
                print(label_folder)
                structures = {'final':'final.in'}
                if HT == True and self.take_HT(label_folder):
                    print('    ---> Select HT structure for DFT SPC')
                    structures['hot'] = 'hot.in'
                self.generate_DFT_data(
                                       label_folder,
                                       structures,
                                       )
        else:
            for label_folder in self.folders.items():
                print(label_folder)
                structures = {'final':'final.in'}
                if HT == True and self.take_HT(label_folder):
                    print('    ---> Select HT structure for DFT SPC')
                    structures['hot'] = 'hot.in'
                self.generate_DFT_data(
                                       label_folder,
                                       structures,
                                       )


    def take_HT(
                self,
                label_folder,
                ):

        label, folder = label_folder
        training = []
        for s in self.training_set:
            if s.info['structure_info'] == label:
                training.append(s)
        final = ase.io.read(os.path.join(folder,'final.in'))
        final = ga.similarity.update_structure(final)
        training.append(final)

        hot = ase.io.read(os.path.join(folder,'hot.in'))
        hot = ga.similarity.update_structure(hot)

        similarity = ga.similarity.structure2training(
                                                      hot,
                                                      training,
                                                      )

        score = np.max(np.transpose(similarity.values()),axis=1).min()
        print('%s: High temperature structure kappa: %s'%(label,np.round(score,3)))
        if score > self.kappa_min:
            return True
        else:
            return False


    def generate_DFT_data(
                          self,
                          label_folder,
                          structures = {
                                        'final':'final.in',
                                        'hot':'hot.in',
                                        },
                          ):
        label, folder = label_folder
        ls = label.split('-')
        for n,s in structures.items():
            if len(ls) == 3:
                tpath = os.path.join(self.dftdir,ls[0],ls[1],ls[2],n)
            else:
                tpath = os.path.join(self.dftdir,ls[0],ls[1],n)
            try:
                pathlib.Path(tpath).mkdir(parents=True)
            except OSError:
                pass
            if n == 'hot':
                QEgenerator.data['calculation'] = 'scf'
            elif 'final' in n:
                QEgenerator.data['calculation'] = 'relax'
                QEgenerator.data['nstep'] = self.geoopt_maxsteps
            atom = ase.io.read(os.path.join(folder,s))
            pw_in_file = os.path.join(tpath,'input.inp')
            pw_in = open(pw_in_file, 'w')
            ase.io.espresso.write_espresso_in(
                                              pw_in,
                                              atom,
                                              input_data=QEgenerator.data,
                                              pseudopotentials=QEgenerator.pseudo,
                                              kspacing=0.05,
                                              koffset=(0,0,0),
                                              )
            pw_in.close()
            self.get_rid_of_K_in_z(pw_in_file)

            ase.io.write(os.path.join(tpath,self.dft_files['input_structure']),atom)
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


    def randdft(
                self,
                ):
        from random import randint
        keeper = []
        for label, folder in self.dft_folders.items():
            s = ase.io.read(os.path.join(folder['final'],self.dft_files['input_structure']))
            result = process_ase(s,
                                 os.path.join(folder['final'],'out'),
                                 )
            s = result[0]
            i = randint(1,19)
            a = s[i]
            a.info['structure'] = 'slab'
            a.info['structure_info'] = label
            keeper.append(a)
        ase.io.write('random.xyz', keeper)


    def extract_forces(
                       self,
                       vstraining = False,
                       HT = True,
                       ):
        self.dft_scores = {}
        keeper = []
        for label, folder in self.dft_folders.items():
            print('\nChecking DFT results of %s'%label)
            keeper += self.check_dft(
                                     folder['final'],
                                     label = label,
                                     vstraining=vstraining,
                                     )
            if HT == True:
                print('\nLooking for HT structure...')
                try:
                    keeper += self.check_dft(
                                             folder['hot'],
                                             label = label,
                                             vstraining=vstraining,
                                             HT = True,
                                             )
                except KeyError:
                    print('No HT structure available')
        self.add_forces = keeper
        add_forces = []
        for s in self.add_forces:
            for x in ['SOAP','structural_info']:
                try:
                    del s.arrays[x]
                except:
                    pass
            add_forces.append(s)
        ase.io.write('add_forces.xyz',add_forces)


    def extract_test_sets(
                       self,
                       ):
        keeper = []
        for label, folder in self.dft_folders.items():
            print('\nChecking DFT results of %s'%label)
            keeper += self.check_dft_for_test(
                                     folder['final'],
                                     label = label,
                                     )
        self.test_add_forces = keeper
        test_add_forces = []
        for s in self.test_add_forces:
            for x in ['SOAP','structural_info']:
                try:
                    del s.arrays[x]
                except:
                    pass
            test_add_forces.append(s)
        ase.io.write('test_add_forces.xyz',test_add_forces)


    def check_dft_for_test(
                  self,
                  folder,
                  label,
                  ):

        from random import randrange, randint

        if not all([os.path.isfile(os.path.join(folder,f)) for f in self.dft_files.values()]):
            print('DFT files not found')
            return []
        if not self.qe_finished(os.path.join(folder,self.dft_files['output'])):
            print('DFT not yet done')
            return []
        s = ase.io.read(os.path.join(folder,self.dft_files['input_structure']))
        result = process_ase(s,
                             os.path.join(folder,'out'),
                             )
        s = result[0]
        s = ga.similarity.force_parsing(s)
        # print(len(s))
        for k in s:
            k.info['structure'] = 'slab'
            k.info['structure_info'] = label
        specific_training = []
        min_energy_training = 1000

        try:
            test_set = ase.io.read('test_set.xyz', ':')
            test_set = ga.similarity.update_structure(test_set)

            for t in test_set:
                if t.info['structure_info'] == label:
                    specific_training.append(t)
                    if t.info['dft_energy']/len(t) < min_energy_training:
                        min_energy_training = t.info['dft_energy']/len(t)
        except:
            specific_training = []

        if len(specific_training) == 0:
            print('Found no same label structure in test set')
            if len(s) == 0:
                return []
            s = ga.similarity.update_structure(s)
            n = randint(1,len(s)-2)
            return [s[n]]

        s = ga.similarity.update_structure(s)
        scores2test = []
        test_candidates = []
        print(len(s[1:-1]))
        for snap in s[1:-1]:
            score = ga.similarity.calculate_score(ga.similarity.structure2training(snap,specific_training))
            if score > 0.075:
                test_candidates.append(snap)
                scores2test.append(score)

        if len(test_candidates) == 0:
            return []
        else:
            n = randrange(len(test_candidates))
            fs = test_candidates[n]
            print('%s: Similarity score from test set.\nScore: %s'%(label, np.round(scores2test[n],3)))

            return [fs]

    def check_dft(
                  self,
                  folder,
                  label,
                  vstraining = False,
                  HT = False,
                  ):
        if not all([os.path.isfile(os.path.join(folder,f)) for f in self.dft_files.values()]):
            print('DFT files not found')
            return []
        if not self.qe_finished(os.path.join(folder,self.dft_files['output'])):
            print('DFT not yet done')
            return []
        s = ase.io.read(os.path.join(folder,self.dft_files['input_structure']))
        result = process_ase(s,
                             os.path.join(folder,'out'),
                             )
        s = result[0]
        for k in s:
            k.info['structure'] = 'slab'
            k.info['structure_info'] = label
        if HT:
            print('Found HT structure for %s'%label)
            return s
        ase.io.write(os.path.join(folder,'forces.xyz'),s)
        ase.io.write(os.path.join(folder,'final.in'),s[-1])
        specific_training = []
        min_energy_training = 1000
        for t in self.training_set:
            if t.info['structure_info'] == label:
                specific_training.append(t)
                if t.info['dft_energy']/len(t) < min_energy_training:
                    min_energy_training = t.info['dft_energy']/len(t)
        
        if len(specific_training) == 0:
            print('Found no same label structure in training set')
            s = ga.similarity.update_structure(s)
            similarity = ga.similarity.structure2training(s[-1],[s[0]])
            score = ga.similarity.calculate_score(similarity)
            if score < self.kappa_min:
                print('--> include final dft structure.\nScore: %s'%(np.round(score,3)))
                return [s[-1]]
            print('--> include initial and final dft structures.\nScore: %s'%(np.round(score,3)))
            return [s[0],s[-1]]

        if s[-1].info['dft_energy']/len(s[-1]) < min_energy_training:
            print(colored('FOUND NEW LOW ENERGY BASIN', 'green'))
        s = ga.similarity.update_structure(s)
        scores = []
        scores2training = []
        energies = []
        for snap in s:
            scores.append(ga.similarity.calculate_score(ga.similarity.structure2training(snap,[s[0]])))
            energies.append(snap.info['dft_energy']/len(snap))
            if vstraining:
                scores2training.append(ga.similarity.calculate_score(ga.similarity.structure2training(snap,specific_training)))
        self.dft_scores[label] = {
                                  'initial':scores,
                                  'training':scores2training,
                                  }

        plot_energy_similarity(
                               scores,
                               energies,
                               training_energy = min_energy_training,
                               output_name = os.path.join(folder,'geoopt_similiarity_energy'),
                               soaplim = (int(np.max(scores)*100) + 2)/100.
                               )
        if vstraining:
            plot_energy_similarity(
                                   scores2training,
                                   energies,
                                   training_energy = min_energy_training,
                                   output_name = os.path.join(folder,'geoopt_similiarity2t_energy'),
                                   soaplim = (int(np.max(scores2training)*100) + 2)/100.
                                   )
        if ga.similarity.different_upper_lower(s[-1]):
            fs = s[-1]
            upper = fs[fs.positions[:,2] > fs.cell[2,2]/2]
            lower = fs[fs.positions[:,2] < fs.cell[2,2]/2]
            us = ga.similarity.structure2training(upper,specific_training)
            ls = ga.similarity.structure2training(lower,specific_training)
            uscore = ga.similarity.calculate_score(us)
            lscore = ga.similarity.calculate_score(ls)
            print('upper')
            #print(uscore)
            ase.io.write(os.path.join(folder,'upper_best_match.in'),ga.similarity.best_match(us,specific_training))
            print('lower')
            #print(lscore)
            ase.io.write(os.path.join(folder,'lower_best_match.in'),ga.similarity.best_match(ls,specific_training))
        similarity = ga.similarity.structure2training(s[-1],specific_training)
        score = ga.similarity.calculate_score(similarity)
        best_match = ga.similarity.best_match(similarity,specific_training)
        ase.io.write(os.path.join(folder,'best_match.in'),best_match)
        if score < self.kappa_min:
            print('%s: Final DFT structure already in training set.\nScore: %s'%(label,np.round(score,3)))
            return [s[0]]
        similarity = ga.similarity.structure2training(s[-1],specific_training + [s[0]])
        if ga.similarity.calculate_score(similarity) < self.kappa_min:
            print('%s: Initial and final DFT structure similary --> keep final'%label)
            return [s[-1]]
        return [s[0],s[-1]]


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
        plot_relative_DFT_energies(results)


    def get_relative_GAP_energies(
                                  self,
                                  labels = [],
                                  ):
        results = pd.DataFrame()
        for snapshot in ['initial','final']:
            surface_areas = []
            labels = []
            energies = []
            for l,f in self.folders.items():
                s = ase.io.read('%s/final.in'%f)
                surface_areas.append(2 * np.linalg.norm(np.cross(s.cell[0],s.cell[1])))
                e = self.md_energies[self.md_energies.label == l][snapshot].values[0]
                for el,ae in self.atomic_energies.items():
                     noa = np.sum(np.array(s.get_chemical_symbols()) == el)
                     e -= noa*ae
                e /= len(s)
                energies.append(e)
                labels.append(l)
            results['label'] = labels
            results['surface_area'] = surface_areas
            results['GAP_' + snapshot] = energies
        if hasattr(self, 'relative_energies'):
            self.relative_energies = self.relative_energies.merge(results,on='label')
        else:
            self.relative_energies = results


    def check_with_old_complexion(
                                  self,
                                  reference='',
                                  ):
        if reference == '':
            raise IOError('reference including old complexions not found')
        reference = ase.io.read(reference,':')
        if 'SOAP' not in reference[0].info:
            reference = ga.similarity.update_structure(reference)
        scores = {}
        new_complexions = []
        for rs in reference:
            label = rs.info['structure_info']
            ts = []
            for s in self.training_set:
                if s.info['structure_info'] == label:
                   ts.append(s)
            sim = ga.similarity.structure2training(rs,ts)
            try:
                best_match = ga.similarity.best_match(
                                                      sim,
                                                      ts,
                                                      )
                ase.io.write('bm_%s.in'%label,best_match)
                ase.io.write('ref_%s.in'%label,rs)
                scores[label] = np.max(np.transpose(sim.values()),axis=1).min()
                new_complexions.append(best_match)
            except ValueError:
                print('No structure for %s found in given trajectory'%label)
        ase.io.write('current_complexions.xyz',new_complexions)
        return scores


def plot_scores_vs_old(scores):
    plt.plot(scores.values(),color='black',marker='x',linestyle='')
    plt.xticks(np.arange(len(scores)),scores.keys(),rotation='vertical',)
    plt.subplots_adjust(bottom=0.20,left=0.15)
    plt.xlim(-1,len(scores))
    plt.ylim(0)
    plt.ylabel(r'$\kappa$(new,old)')
    plt.savefig('similarity2old_complexions')
    plt.close()


def plot_relative_DFT_energies(
                               results,
                               ):

    results['dft_final'].plot(color='blue',label='relaxed',marker='o',linestyle='')
    results['dft_initial'].plot(color='red',label='bulk-truncated',marker='x',linestyle='')
    if 'reference' in results:
        results['reference'].plot(color='black',label='reference',marker='d',linestyle='')
    results['dft_complexion'].plot(color='pink',label='complexion',marker='*',linestyle='')
    plt.legend()
    plt.xticks(
              np.arange(results.shape[0]),
              results['label'],
              rotation='vertical',
              )
    plt.subplots_adjust(bottom=0.30,left=0.15)
    plt.xlim(-1,len(results))
    plt.savefig('relative_energies',dpi=300)
    plt.close()


def plot_GAPvsDFT_energies(
                           results,
                           output_name='GAPvsDFT_formation_energies',
                           sfe=False,
                           ):
    colors={'initial':'red','final':'blue'}
    labels={'initial':'bulk-truncated','final':'relaxed'}
    for snapshot in ['initial','final']:
        x = results['dft_%s'%snapshot]
        y = results['GAP_%s'%snapshot]
        if sfe:
            x /= results['surface_area']
            y /= results['surface_area']

        plt.plot(
                 results['dft_%s'%snapshot],
                 results['GAP_%s'%snapshot],
                 marker = 'o',
                 linestyle = '',
                 color = colors[snapshot],
                 mfc='none',
                 label = labels[snapshot],
                 markersize=10,
                 markeredgewidth=2,
                )
    plt.ylabel(
               r'$ E^{\mathrm{GAP}}_{\mathrm{coh}} (\mathrm{eV}/\mathrm{atom})$',
               fontsize='16',
               )
    ax = plt.gca()
    plt.xlabel(
               r'$ E^{\mathrm{DFT}}_{\mathrm{coh}} (\mathrm{eV}/\mathrm{atom})$',
               fontsize='16',
               )
    ylim = plt.ylim()
    xlim = plt.xlim()
    ll = np.min([xlim[0],ylim[0]])
    hl = np.max([xlim[1],ylim[1]])
    plt.xlim(ll,hl)
    plt.ylim(ll,hl)
    plt.plot([ll,hl],[ll, hl],color='black',linestyle='--')
    plt.legend(fontsize='12')
    plt.subplots_adjust(left=0.22,bottom=0.15)
    plt.savefig(output_name,dpi=300)
    plt.close()


def plot_energy_difference(
                           results,
                           output_name='GAP_DFT_energy',
                           sfe=False,
                           ):
    colors={'initial':'red','final':'blue'}
    labels={'initial':'bulk-truncated','final':'relaxed'}
    # results = update_labels(results)

    fig, ax1 = plt.subplots()

    lns = []
    for snapshot in ['initial','final']:
        x = results['dft_%s'%snapshot]
        y = results['GAP_%s'%snapshot]
        if sfe:
            x /= results['surface_area']
            y /= results['surface_area']
        results['diff_%s'%snapshot] = x - y
        tlns = ax1.plot(
                        results['diff_%s'%snapshot].abs()*1000,
                        marker = 'x',
                        linestyle = '',
                        color = colors[snapshot],
                        mfc='none',
                        label = labels[snapshot],
                        markersize=10,
                        markeredgewidth=2,
                       )
        lns.append(tlns)
    plt.legend(fontsize='12',loc='upper left')
    ax1.set_ylabel(
                   r'$|\gamma_{\mathrm{surf}}^{(hkl),\sigma}|$ $(\mathrm{meV}/\mathrm{\AA}^2)$',
                   fontsize='16',
                   )
    plt.xticks(
               np.arange(results.shape[0]),
               results['label-fancy'],
               rotation='vertical',
                )
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\kappa_{\mathrm{DFT,GAP}}$',fontsize='16')
    tlns = ax2.plot(
                    results['similarity_score'],
                    color = 'green',
                    label = r'$\kappa_{\mathrm{DFT,GAP}}$',
                    linestyle = '',
                    marker = 'd',
                    )
    lns = lns[0] + lns[1] + tlns
    labs = [l.get_label() for l in lns]
    ax2.set_ylim(0,0.2)

    ax1.legend(lns, labs, loc=0,fontsize='12')
    plt.subplots_adjust(left=0.12,bottom=0.30,right=0.85)
    plt.savefig('difference_' + output_name,dpi=300)
    plt.close()




def plot_energy_similarity(
                           similarities,
                           energies,
                           training_energy = 0,
                           output_name = 'geoopt_similiarity_energy',
                           soaplim = 0.25
                           ):

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r'Snapshot $i$')
    ax1.set_ylabel(r'$E_{\mathrm{rel}}$ ($\gamma_{\mathrm{surf}}^{(hkl),\sigma}) \ (\mathrm{meV}/\mathrm{\AA}^2)$')
    ax1.plot(
             energies,
             color = 'black',
             label = r'$E_{\mathrm{rel}}$',
             )
    if training_energy:
        plt.axhline(
                    training_energy,
                    linestyle=':',
                    color = 'grey',
                    linewidth = 1.5,
                    )
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\kappa(i,C_k^{(hkl)-\sigma}$')
    ax2.plot(
             similarities,
             color = 'green',
             label = r'$\kappa_{\mathrm{i,GAP_{min}}}$',
             linestyle = '--',
             )
    ax2.set_ylim(0,soaplim)
    ax1.set_xlim(0,len(energies))
    plt.subplots_adjust(right=0.85)
    plt.subplots_adjust(left=0.15)
    plt.savefig(output_name,dpi=300)
    plt.close()



