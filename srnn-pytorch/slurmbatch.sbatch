#!/bin/bash
#SBATCH -JSocialNavtest
#SBATCH -A gts-awu36-paid
#SBATCH -N2 --gres=gpu:2 --gres-flags=enforce-binding --mem-per-gpu=12G
#SBATCH -t2400
#SBATCH -qinferno
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=splion360@gmail.com
#SBATCH -oSocialNavTest_G1_old_gl6.out

cd ~/p-awu36-0/social_navigation/srnn-pytorch
module load pytorch

# Experiment Order
# G1 experiments
srun python srnn/train.py --train True --save_dir ./saved_models/G1/old/gl6/ --gl --bodyline_only --exp_name G1_old_gl6 --num_epochs 100
# srun python srnn/train.py --train True --save_dir ./saved_models/G1/old/gl12/ --gl --exp_name G1_old_gl12 --num_epochs 100

# srun python srnn/train.py --train True --save_dir ./saved_models/G1/new/gl6/ --gl --bodyline_only --exp_name G1_new_gl6 --num_epochs 100
# srun python srnn/train.py --train True --save_dir ./saved_models/G1/new/gl12/ --gl --exp_name G1_new_gl12 --num_epochs 100

# #G2 experiments

# srun python srnn/train.py --train True --save_dir ./saved_models/G2/old/gl6/ --gl --bodyline_only --exp_name G2_old_gl6 --all_kp --num_epochs 100
# python srnn/train.py --train True --save_dir ./saved_models/G2/old/gl12/ --gl --exp_name G2_old_gl12 --num_epochs 100
 
# srun python srnn/train.py --train True --save_dir ./saved_models/G2/new/gl6/ --gl --bodyline_only --exp_name G2_new_gl6 --all_kp --num_epochs 100
# srun python srnn/train.py --train True --save_dir ./saved_models/G2/new/gl12/ --gl --exp_name G2_new_gl12 --num_epochs 100

