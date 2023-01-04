#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=magic-point-ir_shapes
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:4
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/magic-point-ir_shapes_v1_p100.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python experiment.py train configs/magic-point-ir_shapes.yaml magic-point-ir_shapes_p100

