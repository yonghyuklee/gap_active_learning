import pandas as pd
import numpy as np
import argparse
import ase
import ase.io
import os
from xyz2dump import *

def xyz2data(
             structure,
             vacuum_layer = 10,
             filename = 'slab.data',
             slab = False,
             qO = -0.82,
             ):
    elements = extract_elements(
                                structure,
                                qO,
                                )
    xyz = structure.positions
    symbols = structure.get_chemical_symbols()

    noa = structure.get_global_number_of_atoms()
    at = np.unique(symbols)
    for e in at:
        if e not in elements.keys():
            raise KeyError('No atom_type_id and mass information for %s'%e)


    if not slab:
        lower = np.array([xyz[:,0].min(),xyz[:,1].min(),xyz[:,2].min()])
        upper = np.array([xyz[:,0].max(),xyz[:,1].max(),xyz[:,2].max()])

        # Put vacumm_layer
        lower -= vacuum_layer
        upper += vacuum_layer

        lower = np.floor(lower)
        upper = np.ceil(upper)

        xyz -= lower
        upper -= lower
        lower -= lower

    elif slab:
        cell = structure.cell
        if abs(np.sum(np.triu(cell,k=1))) > 10**-7:
            fn = '%s_re.in'%os.path.splitext(filename)[0]
            print('Triclinic cell is not compartible with LAMMPS')
            print('Rewrite box and store as %s'%fn)
            structure, parameters = xyz2lammpscell(structure)
            ase.io.write(fn,structure)
            cell = structure.cell
        xyz = structure.positions
        lower = np.array([0,0,0])
        upper = cell.diagonal()
        xy,xz,yz = cell[1,0],cell[2,0],cell[2,1]
    file = open(filename,'w')
    file.write('# generated via xyz2data.py (works only for M-Qeq input)\n\n')
    file.write('%s atoms\n'%noa)
    file.write('%s atom types\n\n'%len(at))
    file.write('%s  %s  xlo xhi\n'%(lower[0],upper[0]))
    file.write('%s  %s  ylo yhi\n'%(lower[1],upper[1]))
    file.write('%s  %s  zlo zhi\n'%(lower[2],upper[2]))
    if slab:
        file.write('%s  %s  %s xy xz yz\n'%(xy, xz, yz))
    file.write('\n\n Masses\n\n')
    for e in at:
        file.write('%s %s # %s\n'%(elements[e][0],elements[e][1],e))

    file.write('\n\n Atoms\n\n')

    for a in range(noa):
        file.write('%4i %3i %10.9f %12.8f %12.8f %12.8f\n'%(
                                          a+1,
                                          elements[symbols[a]][0],
                                          elements[symbols[a]][2],
                                          xyz[a][0],
                                          xyz[a][1],
                                          xyz[a][2],
                                          )
                  )

    file.close()

def extract_elements(
                     structure,
                     qO,
                     ):
    noh = sum(np.array(structure.get_chemical_symbols()) == 'H')
    noo = sum(np.array(structure.get_chemical_symbols()) == 'O')
    nozr = sum(np.array(structure.get_chemical_symbols()) == 'Zr')

    qH = - 0.5 * qO
    total_charge = noo * qO - noh *qH
    if nozr != 0:
        qZr = - total_charge / nozr
        elements = {
                    'Zr': [1, 91.224, qZr],
                    'O' : [2, 15.9994, qO],
                    'H' : [3, 1.008,  qH],
                    }
    elif nozr == 0:
        elements = {
                    'O' : [1, 15.9994000,  qO],
                    'H' : [2, 1.008,  qH],
                    }

    #print(elements)
    return elements



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('structure', type=str,
                        help='enter structure here (ASE compatible)')
    parser.add_argument('-vl','--vacuum_layer', type=float, default=10,
                        help='manually define vacuum layer')
    parser.add_argument('-fn','--filename', type=str, default='nanoparticle.data',
                        help='manually define output filename')
    parser.add_argument('-s','--slab', action='store_true',
                        help='generates slab cell')
    parser.add_argument('-c','--cell', type=float, nargs='+',
                        help='pecify box size in format x_min x_max y_min y_max z_min z_max')
    parser.add_argument('-foc','--fixed_oxygen_charge', type=float, default = -0.82,
                        help='Specify qo which will determine remaining charges',
                        )

    args = parser.parse_args()
    structure = ase.io.read(args.structure)
    if args.cell:
        cell = np.zeros([3,3])
        cell[0,0] = args.cell[1]-args.cell[0]
        cell[1,1] = args.cell[3]-args.cell[2]
        cell[2,2] = args.cell[5]-args.cell[4]
        structure.cell = cell
        slab = True
    else:
        slab = args.slab

    xyz2data(
             structure,
             vacuum_layer = args.vacuum_layer,
             filename = args.filename,
             slab = slab,
             qO = args.fixed_oxygen_charge
             )
