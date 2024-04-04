import ase, argparse
import numpy as np 
import ase.io
import pandas as pd 

symbol2type = {'Zr':1,'O':2,'H':3}

def lattice2lammps(
                   cell,
                   ):
    a,b,c = ase.geometry.cell_to_cellpar(cell)[:3]
    angle = ase.geometry.cell_to_cellpar(cell)[3:]
    alpha, beta, gamma = angle/180*np.pi
    lx = a
    xy = b * np.cos(gamma)
    xz = c * np.cos(beta)
    ly = np.sqrt(b**2 - xy**2)
    yz = (b*c*np.cos(alpha) - xy*xz)/ly
    lz = np.sqrt(c**2 - xz**2 -yz**2)
    llat = [lx,ly,lz,xy,xz,yz]
    return llat

def lammps2lattice(
                   llat,
                   ):
    lx,ly,lz,xy,xz,yz = llat
    a = lx
    b = np.sqrt(ly**2 + xy**2)
    c = np.sqrt(lz**2 + xz**2 + yz**2)
    alpha = np.arccos((xy * xz + ly * yz)/(b*c))
    beta = np.arccos(xz/c)
    gamma = np.arccos(xy/b)
    alpha,beta,gamma = np.array([alpha,beta,gamma]) * 180 /np.pi
    return a,b,c,alpha,beta,gamma


def xyz2lammpscell(
                   atom,
                   ):
        cell = atom.cell
        xlo, ylo, zlo = 0, 0, 0
        llat = lattice2lammps(cell)

        xhi, yhi, zhi, xy, xz, yz = llat
        xlo += np.min([0.0,xy,xz,xy+xz])
        xhi += np.max([0.0,xy,xz,xy+xz])
        ylo += np.min([0.0,yz])
        yhi += np.max([0.0,yz])

        parameters = xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz
        newcell = ase.geometry.cellpar_to_cell(lammps2lattice(llat))

        Xs = atom.get_scaled_positions()
        newpositions = np.dot(Xs,newcell)
        a = ase.Atoms(
                      positions=newpositions,
                      symbols=atom.get_chemical_symbols(),
                      pbc=True,
                      cell=newcell
                      )
        return a, parameters 

def write_lammps_dump(
                      trajectory_file,
                      cell,
                      output_file='tmp.lammpstraj',
                      triclinic = False,
                      ):
    atoms = ase.io.read(trajectory_file,index=':')
    try:
        cellfile = open(cell,'r')
        cellfile.readline()
        cf = True
    except:
        cell = cell
        cf = False
    if np.round(cell.sum() - np.diag(cell).sum(),5) != 0.:
        print('Detect triclinic cell --> triclinic dumpfile')
        triclinic = True
    tmp = open(output_file,'w')
    for n,atom in enumerate(atoms):
        noa = atom.get_number_of_atoms()
        if cf == True:
            c = cellfile.readline().split()
            c = np.array([float(x) for x in c][2:-1])
            c = c.reshape([3,3])
        else:
            c = cell
        tmp.write('ITEM: TIMESTEP\n%s\n'%n)
        tmp.write('ITEM: NUMBER OF ATOMS\n%s\n'%noa)
        if triclinic:

            atom, parameters = xyz2lammpscell(atom)
            ase.io.write('atom.in',atom)
            xlo,xhi,ylo,yhi,zlo,zhi,xy,xz,yz = parameters

            tmp.write('ITEM: BOX BOUNDS xy xz yz pp pp pp\n')
            tmp.write('%s %s %s\n'%(xlo,xhi,xy))
            tmp.write('%s %s %s\n'%(ylo,yhi,xz))
            tmp.write('%s %s %s\n'%(zlo,zhi,yz))
        else:
            tmp.write('ITEM: BOX BOUNDS pp pp pp\n')
            tmp.write('0 %s\n0 %s\n0 %s\n'%(c[0,0],c[1,1],c[2,2]))
            xlo, xhi = 0, np.max(cell[:,0])
            ylo, yhi = 0, np.max(cell[:,1])
            zlo, zhi = 0, np.max(cell[:,2])
        tmp.write('ITEM: ATOMS id type x y z ix iy iz\n')
        tdf = pd.DataFrame()
        tdf['Atom_type'] = atom.get_chemical_symbols()
        tdf['X'] = atom.positions[:,0]
        tdf['Y'] = atom.positions[:,1]
        tdf['Z'] = atom.positions[:,2]
        tdf['id'] = tdf.index + 1

        tdf['ix'] = 0
        tdf.loc[tdf.X<xlo,'ix'] = -1
        tdf.loc[tdf.X<xlo,'X'] += xhi

        tdf['iy'] = 0
        tdf.loc[tdf.Y<ylo,'iy'] = -1
        tdf.loc[tdf.Y<ylo,'Y'] += yhi

        tdf['iz'] = 0
        tdf.loc[tdf.Z<zlo,'iz'] = -1
        tdf.loc[tdf.Z<zlo,'Z'] += zlo

        for a,b in symbol2type.iteritems():
            tdf.loc[tdf.Atom_type == a,'Atom_type'] = b
        
        tdf.to_csv(tmp,
                   header=None,
                   index=None,
                   sep=' ',
                   columns=['id','Atom_type','X','Y','Z','ix','iy','iz'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('structure', type=str, 
                        help='enter structure here (ASE compatible)')
    parser.add_argument('-cf', '--cell_file', type=str, 
                        help='enter cell file')
    parser.add_argument('-c','--cell',type=float, nargs='+',
                        help='specify box size in format x_min x_max y_min y_max z_min z_max')
    parser.add_argument('-of','--output_file',type=str, 
                        default = 'lmp_trj.lammpstrj',
                        help='output file name')
    parser.add_argument('-tri','--triclinic',action='store_true',
                        help = 'Force triclinic writing style even for non-triclinc cells'
                        )
    args = parser.parse_args()
    
    try:
        t = open(args.cell_file)
        write_lammps_dump(
                          trajectory_file = args.structure,
                          cell = args.cell_file,
                          output_file = args.output_file,
                          ) 
    except:
        try:
            cell = np.zeros([3,3])
            cell[0,0] = args.cell[1]-args.cell[0]
            cell[1,1] = args.cell[3]-args.cell[2]
            cell[2,2] = args.cell[5]-args.cell[4]
        except:
            traj = ase.io.read(args.structure)
            cell = traj.cell
        write_lammps_dump(
                          trajectory_file = args.structure,
                          cell = cell,
                          output_file = args.output_file,
                          triclinic = args.triclinic,
                          )
            

