#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --job-name=magicpoint_test1oldtf
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/magicpoint_test1oldtf.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint1
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python experiment.py train configs/magic-point_test.yaml magicpoint_test1
