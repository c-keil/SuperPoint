#!/usr/bin/env bash
#SBATCH -p reservation
#SBATCH --reservation=multigpu_test_thamilchelvan.a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=magicpoint_test4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:4
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/magicpoint_test4.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python experiment.py train configs/magic-point_test4.yaml magicpoint_test4
