#!/bin/sh
#BSUB -J version_eval
#BSUB -o version_eval_%J.out
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
cd /zhome/87/9/127623/BachelorProject/cropped_data/Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/performance_metric.py
echo "Done"