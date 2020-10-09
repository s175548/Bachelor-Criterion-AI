#!/bin/sh
#BSUB -J versionAdam
#BSUB -o versionAdam%J.out
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
cd ..
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 DeepLab Adam optim_exp optimizer/Adam
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 MobileNet Adam optim_exp optimizer/Adam
echo "Done"