import ase.io
import ase.io.espresso
import argparse
import os

data = dict({
     'restart_mode':       'from_scratch',
     'prefix':             'BiVO4',
     'calculation':        'scf',
     'outdir':             './tmp/',
     'verbosity':          'high',
     'tstress':            True,
     'tprnfor':            True,
     'etot_conv_thr':      1e-6,
     'forc_conv_thr':      1e-5,
     'nstep':              1,
     'ecutwfc':            90.0,
     'ecutrho':            360.0,
     'occupations':        'smearing',
     'smearing':           'gaussian',
     'degauss':            0.0073498648,
     'mixing_mode':        'local-TF',
     'mixing_beta':        0.7,
     'pseudo_dir':         '/u/ylee/code-backup/QE/pseudo/from_wennie',
     'diagonalization':    'david',
     'conv_thr':           1e-06,
     'wf_collect':         True,
     'disk_io':            'low',
     'startingwfc':        'atomic+random',
     'electron_maxstep':   500,
    })

pseudo = dict({'O': 'O_ONCV_PBE-marton.upf', 'Bi': 'Bi_ONCV_PBE-marton.upf', 'V': 'V_ONCV_PBE-marton.upf'})
kspacing = 0.05
