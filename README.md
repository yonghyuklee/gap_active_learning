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

Tested versions of requirements:

```
ase==3.23.0
dscribe==2.1.1
matplotlib==3.9.2
numpy==2.0.2
pandas==2.2.2
python==3.12.6
scikit-learn==1.5.2
scipy==1.14.1
termcolor==2.4.0
```

# Installation

If your machine has Git installed, simply clone the repo to your local directory by:

```
git clone https://github.com/yonghyuklee/gap_active_learning.git
```

After fetching the gap_active_learning repo, add it to your PYTHONPATH by:

```
export PYTHONPATH=$PYTHONPATH:`pwd`/gap_active_learning
```
Remember to add this export line to your ~/.bashrc or the submission script, so that it is accessible by Python when you run the job.

You need to use the absolute path (you can check it by running `pwd` in Bash shell) for this purpose.

After these, run the following line to test:

```
python -c 'import gap_active_learning'
```
If no error occurs, it should have been imported into your path!


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

**If ValueError encountered**

For example, if you encounter the ValueError such as:
```
ValueError: Array “initial_charges” has wrong shape (60, 1) != (60,).
```
This is not the error from this repository but from ase reading initial_charges within the LAMMPS input file. You can resolve this by editing:
```
path/to/site-packages/ase/io/lammpsrun.py
```
find the line below and revise:
```
   if charges is not None:
       #out_atoms.set_initial_charges(charges) # commend out this line
       out_atoms.set_array('initial_charges', charges, float,charges.shape[1:]) # add this line
```
This issue appears to have been resolved in the latest versions of ASE.
