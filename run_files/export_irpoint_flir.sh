#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=export_irpoint_flir
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/export_ir-point_v2_flir.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python export_detections.py configs/ir-point_flir_export.yaml irpoint_v2 --pred_only --batch_size=64 --export_name=irpoint_v2_flir_export
