#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=ir_point_warp
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:2
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/ir-point_warp_v1.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python experiment.py train configs/irpoint_coco.yaml irpoint_warp_v1 --pretrained_model irpoint_v2
