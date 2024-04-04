import sys
sys.path.insert(1, '/Users/yonghyuk/Dropbox/python/gap_active_learning/')
from generation_analysis.generation import *
from generation_analysis.similarity import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-p','--plot_info', action='store_true',
                        help='Plot information about the DFT GO')
    parser.add_argument('-vst','--vs_training', action='store_true',
                        help='Plot information about the DFT GO vs training')
    args = parser.parse_args()

    self = GapGen('gap.xml','training_set.xyz')
    print('Extracting DFT calculations')
    self.get_dft_folders()
    # print(self.dft_folders)
    self.extract_forces()
    # soaplim =0.6
    # if args.plot_info:
    #    # soaplim = (int(np.max([x['initial'] for x in self.dft_scores.values()])*100) + 1)/100.
    #     plot_scores(self.dft_scores,style='initial',soaplim=soaplim)
    # if args.vs_training:
    #    # soaplim = (int(np.max([x['training'] for x in self.dft_scores.values()])*100) + 1)/100.
    #     plot_scores(self.dft_scores,style='training',soaplim=soaplim,output_name='training_geoopt_score')


