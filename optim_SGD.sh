#!/bin/sh
#BSUB -J versionSGD
#BSUB -o versionSGD%J.out
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
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 DeepLab SGD lr_exp optimizer/SGD
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 MobileNet SGD lr_exp optimizer/SGD
echo "Done"