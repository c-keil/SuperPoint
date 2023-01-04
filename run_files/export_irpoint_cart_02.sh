#!/usr/bin/env bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=export_irpoint_cart_02_desc
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --output=/home/thamilchelvan.a/NUFRL/SuperPoint/logs/export_ir-point_v5_cart02_desc.%j.out
source ~/.bashrc
module load cuda/10.0
conda activate SuperPoint
cd /home/thamilchelvan.a/NUFRL/SuperPoint/superpoint/
python extract_desc.py irpoint_v5 /home/thamilchelvan.a/NUFRL/Data/2021-11-08_IR_cart_24hr/ir_cart_2021-11-09-02-04-07/rgb/
