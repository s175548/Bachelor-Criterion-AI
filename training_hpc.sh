#!/bin/sh
#BSUB -J lr1
#BSUB -o lr1%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=128G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source test-env/bin/activate
python3 Bachelor-Criterion-AI/semantic_segmentation/DeepLabV3/Training_windows.py
echo "Done"