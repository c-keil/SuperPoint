#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=export_magicpoint_coco
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/export_magic-point-sp_v6.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python export_detections.py configs/super-point_coco_export.yaml sp_v6 --pred_only --batch_size=64 --export_name=sp_v6_coco_export
