#!/bin/sh
#BSUB -J versionSemiSupervised01
#BSUB -o versionSemiSupervised01%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16G]"
#BSUB -R "select[gpu16gb]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd /zhome/87/9/127623/BachelorProject
source test-env/bin/activate
cd /zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI
python3 semantic_segmentation/semi_supervised/exp_gen_semi.py 0.003 DeepLab SGD False semi_new_weights semi_super/different_loss_weights True True
echo "Done"