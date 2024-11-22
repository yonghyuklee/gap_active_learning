import argparse
from gap_active_learning.al.vasp import *
from gap_active_learning.al.similarity import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-mlp','--mlp', type=str, default='GAP',
                        help='Choose MLP class among GAP and MACE (default: GAP)')
    args = parser.parse_args()

    if args.mlp == 'GAP':
        self = GapGen('gap.xml','training_set.xyz')
        print('Extracting DFT calculations')
        self.get_dft_folders()
        self.extract_forces()

    elif args.mlp == 'MACE':
        self = MACEGen(
                       md_files = {
                                 'final_structure' :'md.xyz',
                                 },
                       max_selected = 1000,
                       nn_uncertainty = 0.1,
                       max_force = 30,
                      )
        print('Extracting DFT calculations')
        self.get_dft_folders()
        self.extract_forces()