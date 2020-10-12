#!/bin/sh
#BSUB -J versionRes_whole
#BSUB -o versionRes_whole%J.out
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
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 DeepLab SGD True binary_exp binary_vs_multi/binary/ResNet True
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 DeepLab SGD True binary_exp binary_vs_multi/multi/ResNet False
echo "Done"