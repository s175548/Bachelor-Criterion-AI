#!/bin/sh
#BSUB -J version01
#BSUB -o version01%J.out
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/dataloader.py
echo "Done"