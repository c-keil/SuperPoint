#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=export_irpoint_flir_desc
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/export_ir-point_v5_flir_desc.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python extract_desc.py irpoint_v5 /home/thamilchelvan.a/NUFRL/SuperPoint/data/FLIR/train/
