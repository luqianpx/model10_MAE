#!/bin/bash
#BSUB -J gpu_test
#BSUB -n 13
#BSUB -gpu "num=1"
#BSUB -o output_%J
#BSUB -e errput_%J
#BSUB -q 4gpu

cd /share/home/hulianting/Project/Project20_ECG_foundation_model/Model/model10_MAE/
module load anaconda
source activate hlt_py310_torch_new
module load cuda/11.5.0

python main_finetune.py --sa_folder=nomix_1500_1024_2500_H2y1


