import sys
from gap_active_learning.al.qe import *
from gap_active_learning.al.similarity import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-HT','--high_temperature', action='store_true',
                        help='Include HT structures (if available)')
    args = parser.parse_args()

    self = GapGen(
                  'gap.xml',
                  'training_set.xyz',
                   # kappa_min = 0.00000001,
                   # geoopt_maxsteps = 100,
                  )
    print('Calculating best similarities')
    self.similarity_analysis(entire_training_set=True)
    self.determine_selected_slabs(compare2entiretraining = True)
    self.write_selected_for_DFT()
    self.write_final_MD_structures()
    if len(self.selected_folders) == 0:
        print('\n\nNo new candidates found for kappa threshold %s'%self.kappa_min)
        # quit()
    print('\n\n Selected folders:')
    print(self.selected_folders)
    for k in self.selected_folders:
        print('%s with kappa: %s'%(k,calculate_score(self.similarities[k])))
    print('\n Writing DFT data')
    if args.high_temperature:
        self.write_hot()
    self.generate_all_DFT_data(HT=args.high_temperature)
