#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=magicpoint_coco_homographic_v1
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/magicpoint_coco_homographic_v1.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python experiment.py train configs/magic-point_coco_train.yaml magicpoint_coco_homographic_v1
