#!/bin/sh
#BSUB -J versionMNExp
#BSUB -o versionMNExp%J.out
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
cd --
source BachelorProject/test-env/bin/activate
cd BachelorProject/cropped_data/Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/experiment_generator.py 0.01 MobileNet SGD True res_exp original_res/smaller True
echo "Done"