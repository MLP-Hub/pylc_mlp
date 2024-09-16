#!/bin/bash
#SBATCH --account=def-ehiggs
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-node=v100l:1

module load StdEnv/2020
module load gcc/9.3.0
module load python/3.10
module load opencv/4.8.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install -r ../requirements.txt

python ../pylc.py train --db /path/to/database.h5
