#!/bin/bash
#SBATCH -N SocialNavtest
#SBATCH -A gts-awu36-paid
#SBATCH -l nodes=2:ppn=4:gpus=3
#SBATCH -l pmem=2gb
#SBATCH -l walltime=210:00
#SBATCH -q testflight-gpu
#SBATCH -m abe
#SBATCH -M splion360@gmail.com
#SBATCH -j oe
#SBATCH -o SocialNavTest.out

cd ~/p-awu36-0/social_navigation/srnn-pytorch
module load pytorch

srun python srnn/train.py --num_epochs 100 --body_keypoint "0,6,9" 
