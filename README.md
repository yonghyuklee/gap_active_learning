# GAP Active Learning Workflow for Surface Structure Determination

An integrated protocol for iterative training of the Gaussian Approximation Potential, combined with surface structure exploration through global optimization using simulated annealing.

Copyright © 2024 Yonghyuk Lee

# Citation

Please CITE the following paper if you use any methodologies from this repository:

```
@article{Timmermann2021,
  title={Data-efficient iterative training of Gaussian approximation potentials: application to surface structure determination of rutile IrO$_{2}$ and RuO$_{2}$},
  author={Timmermann, Jakob and Lee, Yonghyuk and Staacke, Carsten G and Margraf, Johannes T and Scheurer, Christoph and Reuter, Karsten},
  journal={The Journal of Chemical Physics},
  volume={155},
  number={24},
  year={2021},
  publisher={AIP Publishing}
}
```
The manuscript titled `Machine-Learning-Accelerated Surface Exploration of Reconstructed BiVO4(010) and Characterization of Their Aqueous Interfaces` is currently under peer review, with further details to be provided upon publication. Please CITE this paper if you utilize the training set of BiVO4 surfaces.

# Requirements

`pip install -r requirements.txt`

# Tuturial

**BiVO4**

change the directory to examples and unzip example_BiVO4.zip

```
cd examples
unzip example_BiVO4.zip
```

change the directory to the working directory

```
cd example_BiVO4
```

In this directory, there are the following contents:

• `training_set.xyz`: Extended XYZ file containing all the training structures.

• `job_gap.cmd`: Command file for GAP training, including the hyperparameters.

• `md/`: Directory containing all LAMMPS-based simulated annealing executions.

• `dft/`: Directory containing all Quantum Espresso-based DFT calculations.

After running canonical molecular dynamics in the `md/` directory, the code reads all annealed output structures and applies farthest point sampling (FPS) based on the SOAP kernel distance between the results and the known structures in the training set. If the calculated distance exceeds a preset threshold, the code will initiate constrained DFT geometry relaxations. To start this process, use the following command:

```
python path/to/gap_active_learning/gap_active_learning/runner/qe/analyze_MDs.py
```

All final structures from the MD trajectory are saved in the `final_MD_structures.xyz` file, while the dissimilar structures selected through farthest point sampling (FPS) are saved in the `selected_MD_structures.xyz` file. Additionally, a `dft/` folder is generated, which includes the input files for the DFT optimizations.

Once the DFT calculations are complete in the `dft/` directory, you will need to select structures to add to your training set to improve the GAP. To do this, run the following command:

```
python path/to/gap_active_learning/gap_active_learning/runner/qe/analyze_DFT.py
```

The final selected structures are saved in the `add_forces.xyz` file.
