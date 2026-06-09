#!/bin/bash
#SBATCH --chdir=.
#SBATCH --account=digital_human_jobs
#SBATCH --time=23:00:00
#SBATCH --output=/home/%u/slurm_output__%x-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --gpus=2080ti:1
#SBATCH --mem=24G

set -euo pipefail

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "STARTING AT $(date)"

# System modules are not automatically available in batch jobs.
. /etc/profile.d/modules.sh
module add cuda/13.0
nvidia-smi

cd /work/courses/digital_human/team3/3d_head_stylization

# Activate conda environment if needed
if [ -f "/work/courses/digital_human/team3/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "Activating conda"
    source "/work/courses/digital_human/team3/miniconda3/etc/profile.d/conda.sh"
    conda activate 3d_head
fi

echo "Running training script"

python3 train_LD2.py \
  --prompt "Portrait of a person in Renaissance oil painting style, dramatic Rembrandt lighting, painterly brushstrokes, warm shadows, textured canvas, realistic painted face, museum masterpiece, highly detailed" \
  --save_path "work_dirs/oil_paint" \
  --diff_ckpt_path models/Realistic_Vision_V5.1_noVAE \
  --G_ckpt_path models/easy-khair-180-gpc0.8-trans10-025000.pkl \
  --use_SDS 1 \
  --num_angles 7 \
  --num_pitch_angles 1 \
  --yaw_range_front -1.57 1.57 \
  --resume
  