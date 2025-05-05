import ase, ase.io
import os 
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
from dscribe.descriptors import SOAP
from sklearn.preprocessing import normalize
from ase.neighborlist import NeighborList, natural_cutoffs
from scipy import sparse

def update_structure(
                     traj,
                     species=[],
                     soap_info = {
                                  'rcut':5,
                                  'nmax':8,
                                  'lmax':4,
                                  'sigma': 1.0,
                                  'average': 'off',
                                  }
                     ):

    if not species:
        species = []
        if type(traj) == list:
            for t in traj:
                elements = np.unique(t.get_chemical_symbols())
                for e in elements:
                    if e not in species:
                        species.append(e)
        elif type(traj) == ase.atoms.Atoms:
            elements = np.unique(traj.get_chemical_symbols())
            for e in elements:
                if e not in species:
                    species.append(e)

    soap = SOAP(
                species=species,
                periodic=True,
                r_cut=soap_info['rcut'],
                n_max=soap_info['nmax'],
                l_max=soap_info['lmax'],
                sigma=soap_info['sigma'],
                average=soap_info['average'],
                )

    if type(traj) == list:
        for s in traj:
            s.set_array('SOAP',normalize(soap.create(s)))
    elif type(traj) == ase.atoms.Atoms:
            traj.set_array('SOAP',normalize(soap.create(traj)))

    return traj
    
            
def different_upper_lower(
                          structure,
                          ):
    s = structure
    elements = np.unique(s.get_chemical_symbols())
    upper = s[s.positions[:,2] > s.cell[2,2]/2]
    lower = s[s.positions[:,2] < s.cell[2,2]/2]

    soaps_upper = {}
    soaps_lower = {}
    max_similarity = []
    for e in elements: 
        eu = upper[np.array(upper.get_chemical_symbols()) == e]
        el = lower[np.array(lower.get_chemical_symbols()) == e]
        eus = eu.get_array('SOAP')
        els = el.get_array('SOAP')
        for sv in eus:
            max_similarity.append((np.dot(els,sv)**2).max())

    if np.min(max_similarity) < 0.998:
        #print(np.min(max_similarity))
        return True
    else:
        #print(np.min(max_similarity))
        return False


def splitnmerge(
                structure,
                label,
                centerval = 0.1,
                newval = 0.1,
                ):

    minimize = False
    structure = update_structure(structure)
    cell = structure.cell

    upper = structure[structure.positions[:,2] > (cell[2,2]/2 - centerval)]
    new_lower = structure[structure.positions[:,2] > (cell[2,2]/2 + newval)]
    if len(upper + new_lower) != len(structure):
        print(len(upper + new_lower))
        print(len(structure))
        raise ValueError('Mismatching number of atoms in spliting process')
    lower = structure[structure.positions[:,2] < (cell[2,2]/2 + centerval)]
    new_upper = structure[structure.positions[:,2] < (cell[2,2]/2 - newval)] 
    if len(lower + new_upper) != len(structure):
        print(len(lower))
        print(len(new_upper))
        raise ValueError('Mismatching number of atoms in spliting process')
    new_lower.positions[:,2] = cell[2,2] - new_lower.positions[:,2]
    new_upper.positions[:,2] = cell[2,2] - new_upper.positions[:,2]
    if np.count_nonzero(cell - np.diag(np.diagonal(cell))):
        if '101' in label:
            for s in [new_lower, new_upper]:
                s.rotate(180,'z')
                s.positions[:,0] -= cell[0,0]/7
                s.positions[:,1] += cell[1,1]/5
        elif '111' in label:
            for s in [new_lower, new_upper]:
                s.rotate(180,'z')
               # s.positions[:,0] -= cell[0,0]/7
                s.positions[:,1] -= cell[1,1]/5
        minimize = True
    split = {}
    split['lower'] = update_structure([(lower + new_upper)])[0]
    split['upper'] = update_structure([(upper + new_lower)])[0]
    for l,s in split.items():
        s.wrap()
        ase.io.write('2x%s.in'%(l),s)

    return minimize


def quick_score(
                test_structure,
                training_structure,
                ):
    soap_test = test_structure.arrays['SOAP']
    soap_train = training_structure.arrays['SOAP']
    scores = np.dot(soap_train,soap_test.transpose())**2
    score = np.min(np.max(scores,axis=0))
    v = 2*(1-score)
    if v < 10e-10:
        score = 0 
    else:
        score = np.sqrt(v)
    return score


def compare2structures(
                       s1,
                       s2,
                       species=[],
                       ):
    soap_s2 = {}
    kpis = {}
    if len(species) != 0:
        elements = species
    else:
        elements = np.unique(s1.get_chemical_symbols())

    for e in elements:
        select = s2[np.array(s2.get_chemical_symbols()) == e]
        soap_select = select.get_array('SOAP')
        soap_s2[e] = soap_select
        
    max_similarity = []
    for nn,sv in enumerate(s1.get_array('SOAP')):
        max_similarity.append((np.dot(soap_s2[s1[nn].symbol],sv)**2).max())
    s1.set_array('max_similarity',np.array(max_similarity))
   
    for e in elements:
        test_select = s1[np.array(s1.get_chemical_symbols()) == e]
        soap_test_select = test_select.arrays['max_similarity']
        v = 2*(1-soap_test_select.mean())
        if v < 10e-10:
            kpis['%s_mean'%e] = 0
        else:
            kpis['%s_mean'%e] = np.sqrt(v)
        v = 2*(1-soap_test_select.min())
        if v < 10e-10:
            kpis['%s_max'%e] = 0
        else:
            kpis['%s_max'%e] = np.sqrt(v)
    return s1, kpis


def structure2training(
                       s1,
                       training,
                       species=[],
                       ):
    elements = np.unique(s1.get_chemical_symbols())

    similarity = {}
    for e in elements:
        similarity[e] = np.ones(len(training))

    for n,s2 in enumerate(training):
        if set(elements).issubset(np.unique(s2.get_chemical_symbols())):
            ts1, kpis = compare2structures(
                                           s1,
                                           s2,
                                           species=species,
                                           )
            for e in elements:
                similarity[e][n] = np.round(kpis['%s_max'%e],8)
        else:
            pass

    return similarity


def structures2training(
                       structures,
                       training,
                       species=[],
                       ):
    similarity = {}
    for n1,s1 in enumerate(structures):
        elements = np.unique(s1.get_chemical_symbols())
        for e in elements:
            similarity[e] = np.ones(len(training))

    for n2,s2 in enumerate(training):
        if set(elements).issubset(np.unique(s2.get_chemical_symbols())):
            ts1, kpis = compare2structures(
                                           s1,
                                           s2,
                                           species=species,
                                           )
            for e in elements:
                similarity[e][n2] = np.round(kpis['%s_max'%e],8)
        else:
            pass

    return similarity



def calculate_score(
                    sim,
                    ):
    try:
        score = np.max(np.transpose([*sim.values()]),axis=1).min()
    except:
        score = 0
        for side in ['lower','upper']:
            tsim =  sim[side]
            tscore = np.max(np.transpose([*tsim.values()]),axis=1).min()
            if tscore > score:
                score = tscore
    return score
    
    
def best_match(
               similarity,
               training, 
               ):
    bs = np.max(np.array([*similarity.values()]).transpose(),axis=1)
    print('Best match similarity: %s'%np.round(bs.min(),3))
    return training[bs.argmin()]


def generation_similarity_analysis(
                                   slabs,
                                   training = [],
                                   structure_name = 'final_trajectory.in',
                                   vs = [0.985,0.998],
                                   output_filename = 'similarity',
                                   ):
    mddir = os.getcwd()
    if len(training) == 0:
        training = ase.io.read('%s/../training_set.xyz'%mddir,':')
    training = update_structure(training)
    vmin, vmax = vs

    similarities = {}
    for slab in slabs:
        os.chdir('/'.join(slab.split('-')))
        structure = ase.io.read(structure_name)
        structure = update_structure([structure])[0]
        specific_training = []
        for s in training:
            if s.info['structure_info'] == slab:
                specific_training.append(s)
        ase.io.write('training_%s.xyz'%slab,specific_training)
        if different_upper_lower(structure):
            print('%s: Two different surfaces detected for test structure'%slab)
            split_surface = {}
            split_surface['upper'] = structure[structure.positions[:,2] > structure.cell[2,2]/2]
            split_surface['lower'] = structure[structure.positions[:,2] < structure.cell[2,2]/2]
            for side,atoms in split_surface.items():
                print(side)
                similarity = structure2training(
                                                atoms,
                                                specific_training,
                                                )
                ase.io.write('%s_best_match.in'%side,best_match(similarity,specific_training))
                plot_single(
                            similarity,
                            vmin  = vmin, 
                            vmax = vmax,
                            output_filename = side + '_' + output_filename,
                            )
                split_surface[side] = similarity
            similarities[slab] = split_surface
        else:
            similarity = structure2training(
                                            structure,
                                            specific_training,
                                            )
            ase.io.write('best_match.in',best_match(similarity,specific_training))
            plot_single(
                        similarity,
                        vmin  = vmin, 
                        vmax = vmax,
                        output_filename = output_filename,
                        )
            similarities[slab] = similarity

        os.chdir(mddir)

    return similarities

def force_parsing(
                  slabs,
                  max_force = 30,
                  ):
    if hasattr(slabs, 'ase_objtype'):
        print("Found only one atom object!")
        con = False
        for s in slabs.arrays['dft_forces']:
            for f in s:
                if abs(f) >= 30:
                    con = True
        if con:
            return []
    else:
        print("Found list of atom object!")
        fslabs = []
        for slab in slabs:
            con = False
            for s in slab.arrays['dft_forces']:
                for f in s:
                    if abs(f) >= 30:
                        con = True
            if con:
                pass
            else:
                fslabs.append(slab)
        return fslabs


def examine_unconnected_components(atoms):
    nat_cut = natural_cutoffs(atoms, mult=1.2)
    nl = NeighborList(nat_cut, skin=0, self_interaction=False, bothways=True)
    nl.update(atoms)
    matrix = nl.get_connectivity_matrix()
    n_components, component_list = sparse.csgraph.connected_components(matrix)
    if n_components == 1:
        return True
    elif n_components > 1:
        return False


def plot_single(
                similarity,
                vmin  = 0.989,
                vmax = 0.98,
                output_filename = 'single_soap_analsyis',
                ):
    fig, axs = plt.subplots(1,1)
    im = axs.imshow(np.array([*similarity.values()]),vmin=vmin,vmax=vmax)
    axs.tick_params(axis='both',labelsize=14)
    axs.set_xlabel('Structures',fontsize=16)
    # axs.set_yticks([0,1])
    axs.set_yticks(np.linspace(0,len(similarity)-1,len(similarity)))
    axs.set_yticklabels(similarity.keys())
    axs.tick_params(axis='y', which='both', labelsize=24)
    plt.savefig(output_filename,bbox_inches='tight',dpi=300)
    plt.close()


def plot_generation(
                    new, 
                    vs = [0.985,0.998],
                    output_filename = 'similarity_generation',
                    slabs = []
                    ):
    vmin, vmax = vs
    fig, axs = plt.subplots(len(new),1,)
    if len(slabs) != len(new):
        slabs = new.keys()
    for n,slab in enumerate(slabs):
        axs[n].tick_params(axis='both',labelsize=14)
        try:
            plot_this = new[slab].values()
            im = axs[n].imshow(plot_this,vmin=vmin,vmax=vmax)
            axs[n].set_yticks([0,1])
            axs[n].set_yticklabels(new[slab].keys())
            title = axs[n].set_title(slab)

        except TypeError:
            tnew = new[slab]
            plot_this = tnew['upper'].values() + tnew['lower'].values()
            im = axs[n].imshow(plot_this,vmin=vmin,vmax=vmax)
            axs[n].set_yticks([0,1,2,3])
            axs[n].set_yticklabels(tnew['lower'].keys() + tnew['upper'].keys())
            axd = axs[n].twinx()
            axd.set_yticks([0.25,0.75])
            axd.set_yticklabels(['lower','upper'])
            axd.tick_params(axis='y',length=0)
            title = axs[n].set_title(slab)

    for ax in axs[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xticks([])
    axs[-1].set_xlabel('Training structures',fontsize=16)
    plt.subplots_adjust(right=0.7)
    cbar_ax = fig.add_axes([0.78, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Similarity',fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    xlim = 0
    for k in axs:
        if k.get_xlim()[1] > xlim:
            xlim = k.get_xlim()[1]
    for k in axs:
        k.set_xlim(-0.5,xlim)
    plt.savefig(output_filename,dpi=300)#bbox_inches='tight',dpi=300)
    plt.close()


def plot_cross(
               similarity, 
               vmin  = 100,
               vmax = 1,
               output_filename = 'soap_analsyis',
               ):
    fig, axs = plt.subplots(1,1)
    if vmin > 1.0:
        vmin = np.min([vmin, similarity.min()])
    else:
        vmin = vmin

    axs.tick_params(axis='both',labelsize=14)
    im = axs.imshow(similarity,vmin=vmin,vmax=vmax)
    axs.set_title('Similarity',fontsize=16)
    axs.set_xlabel('Structures',fontsize=16)

    axs.set_ylabel('Structures',fontsize=16)
    fig.subplots_adjust(right=0.7)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Similarity',fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    plt.savefig(output_filename,bbox_inches='tight',dpi=300)
    plt.close()

