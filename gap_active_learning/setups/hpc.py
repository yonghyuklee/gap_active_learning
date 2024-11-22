import os
from pathlib import Path

# modify DFT starter below
def write_dft_starter(dftdir, control_file="dft.cmd", script_name="start_dft.sh", walltime="06"):
    """
    Writes a shell script to start DFT calculations for directories containing POSCAR files.

    Args:
        dftdir (str): Directory where the script will be created.
        control_file (str): The name of the control file (default: "dft.cmd").
        script_name (str): The name of the shell script to create (default: "start_dft.sh").
        walltime (str): Walltime value to substitute in the control file (default: "06").
    """
    # Define the script content
    script_content = f"""#!/bin/bash
curdir=$(pwd)
controlfile=$curdir/{control_file}
directories=$(find . -type f -name "POSCAR" -exec dirname {{}} \\;)
for d in $directories; do
    cd $d
    if ! [ -f stdout ] && ! [ -f queued ]; then
        last_part=$(basename "$d")
        second_part=$(echo "$d" | awk -F'/' '{{print $2}}')
        m="${{second_part}}-${{last_part}}"
        jobname=$m
        echo $m
        cp $controlfile control.cmd
        sed -i "s/JOBNAME/$jobname/" control.cmd
        sed -i "s/WALLTIME/{walltime}/" control.cmd
        qsub control.cmd
        touch queued
    fi
    cd $curdir
done
"""

    # Write the script to the specified file
    script_path = Path(dftdir) / script_name
    with open(script_path, "w") as file:
        file.write(script_content)

    # Make the script executable
    script_path.chmod(0o755)
        

# modify JOB SCRIPT for your system
def write_job_script(
                     dftdir,
                     hpc='Polaris',
                     control_file='dft.cmd',
                    ):
    """
    Writes a job script for a specific HPC system.

    Args:
        dftdir (str): Directory where the job script will be written.
        hpc (str): HPC system ("Polaris", "Hoffman", "DOD").
        control_file (str): Name of the control file (default: "dft.cmd").
    """
    hpc_scripts = {
        "Polaris": """#!/bin/sh
#PBS -l select=25
#PBS -l place=scatter
#PBS -l walltime=WALLTIME:00:00
#PBS -l filesystems=home:grand:eagle
#PBS -q prod
#PBS -N JOBNAME
#PBS -A CatDynEnsemble

source ~/.bashrc
conda activate torch
module purge
module load nvhpc/23.9
module load PrgEnv-nvhpc/8.5.0
module load cray-libsci/23.12.5
module load craype-accel-nvidia80
module load cray-fftw/3.3.10.6

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/extras/qd/lib

# Change to working directory
cd ${PBS_O_WORKDIR}

export MPICH_GPU_SUPPORT_ENABLED=1
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=2
NDEPTH=4
NTHREADS=4
NGPUS=2
NTOTRANKS=$(( NNODES * NRANKS ))

export ASE_VASP_COMMAND="mpiexec -n ${NTOTRANKS} --ppn ${NRANKS} --depth ${NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS=${NTHREADS} /home/ylee/codes/VASP/source/vasp.6.4.1-vtst/bin/vasp_std"
export VASP_PP_PATH=/home/ylee/codes/VASP/source

rm queued
python run_vasp.py
""",

        'Hoffman': """#!/bin/sh -f
#$ -l h_data=3G,h_rt=WALLTIME:30:00
#$ -cwd
#$ -o LOG.$JOB_NAME.$JOB_ID
#$ -j y
#$ -pe dc* 128
#$ -N JOBNAME

export Nprocs=128
echo RUNNING ON $HOSTNAME
cat $PE_HOSTFILE

source /u/local/Modules/default/init/modules.sh
source ~/.bashrc
module purge
module load IDRE intel/2020.4 intel/mpi hdf5/1.12.0_intel2019.2 anaconda3/2020.11
conda activate cupd

export ASE_VASP_COMMAND="mpirun -n $Nprocs /u/home/y/ylee/code/vasp/vasp.6.4.1/vasp.6.4.1/bin/vasp_std"
export VASP_PP_PATH=/u/home/y/ylee/code/vasp/vasp.6.4.1/POTCAR

rm queued
python run_vasp.py
""",

        'DOD': """#!/bin/csh
#PBS -A AFOSR35083MAV
#PBS -q standard
#PBS -l select=NNODES:ncpus=128:mpiprocs=128
#PBS -l walltime=WALLTIME:00:00
#PBS -j oe
#PBS -N JOBNAME

setenv Nprocs NNCPUS
cd $PBS_O_WORKDIR

source ~/.bashrc
bash -c 'source ~/.bashrc && conda activate'
module unload PrgEnv-cray
module load PrgEnv-intel
module load cray-fftw

setenv ASE_VASP_COMMAND "mpiexec -n $Nprocs /p/home/schiu479/VASP/vasp.6.4.1_cpu/bin/vasp_std"
setenv VASP_PP_PATH /p/home/schiu479/VASP/pseudo

rm queued
python run_vasp.py
"""
    }

    # Check if the HPC system is supported
    if hpc not in hpc_scripts:
        raise ValueError(f"Unsupported HPC system: {hpc}")

    # Write the script to the specified directory
    script_path = Path(dftdir) / control_file
    with open(script_path, "w") as file:
        file.write(hpc_scripts[hpc])

    # Make the script executable
    script_path.chmod(0o755)
            

# def write_dft_starter(self):
#         file = open(os.path.join(self.dftdir,'start_dft.sh'),'w')
#         file.write("""curdir=$(pwd)
# controlfile=$curdir/dft.cmd
# for m in `ls -d */`; do
#         cd $m
#         m=$(echo "$m" | tr -d ./)
#         for t in `find . -maxdepth 1 -mindepth 1 -type d -name '*t*'`; do
#                 cd $t
#                 t=$(echo "$t" | tr -d ./)
#                 jobname=$1$m$t
#                 echo $m $t
#                 cd final
#                 cp $controlfile control.cmd
#                 sed -i "s/JOBNAME/$jobname/" control.cmd
#                 sed -i "s/WALLTIME/08/" control.cmd
#                 sbatch control.cmd
#                 cd ../
#                 cd ../
#             done
#         cd ../
# done
#                    """)
#         file = open(os.path.join(self.dftdir,'dft.cmd'),'w')
# # UPDATE FOR YOUR SUBMISSION SCRIPT
#         file.write("""#!/bin/bash -l
# #SBATCH -o ./tjob.out
# #SBATCH -e ./tjob.err
# #SBATCH -D ./
# #SBATCH -J JOBNAME
# #SBATCH --nodes=4
# #SBATCH --ntasks-per-node=40
# #SBATCH --mail-type=all
# #SBATCH --mail-user=
# # Wall clock limit:
# #SBATCH --time=WALLTIME:00:00

# module purge
# unset LD_LIBRARY_PATH
# module load anaconda/3/2021.11 gcc/12 impi/2021.7 mkl/2022.2 gsl/2.4 intel/21.7.1
# source /u/ylee/.bashrc
# conda activate BiVO4
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$MKLROOT/lib/intel64/"
# source "$MKLROOT/env/vars.sh"
# MKLL=$MKLROOT/lib/intel64/libmkl_
# IOMP5=$(realpath "$MKLROOT/../../compiler/2022.2.1/linux/compiler/lib/intel64_lin/libiomp5.so")
# export LD_PRELOAD="${MKLL}def.so.2:${MKLL}avx512.so.2:${MKLL}core.so.2:${MKLL}intel_lp64.so.2:${MKLL}intel_thread.so.2:$IOMP5"

# srun /u/ylee/code-backup/QE/releases/qe-7.0/bin/pw.x -nk 8 < input.inp > out

#                    """)