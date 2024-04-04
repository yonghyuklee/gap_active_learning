import sys
sys.path.insert(1, '/Users/yonghyuk/Dropbox/python/gap_active_learning/')
from generation_analysis.generation import *
from generation_analysis.similarity import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args = parser.parse_args()

    self = GapGen(
                  'gap.xml',
                  'training_set.xyz',
                #   kappa_min = 0.00000001,
                #   geoopt_maxsteps = 100,
                  uncertainty_min = 50,
                  kappa_min = 0.1
                  )
    print('Calculating best similarities')
    self.uncertainty_analysis()
    self.write_selected_for_DFT()
    if len(self.selected_folders) == 0:
        print('\n\nNo new candidates found for kappa threshold %s'%self.kappa_min)
    print('\n\n Selected folders:')
    print(self.selected_folders)
    for k in self.uncertainties:
        print("Maximum uncertanty for {} is {} meV/atom".format(k, self.uncertainties[k]))
    print('\n Writing DFT data')
    self.generate_all_DFT_data()
