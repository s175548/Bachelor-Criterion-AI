#!/bin/sh
#BSUB -J versionPre_trained_setup
#BSUB -o versionPre_trained%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd /zhome/87/9/127623/BachelorProject
source test-env/bin/activate
cd /zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 DeepLab SGD False pre_trained semi_super/no_pre_train True False
echo "Done"