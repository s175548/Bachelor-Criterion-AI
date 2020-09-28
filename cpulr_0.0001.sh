#!/bin/sh
#BSUB -J cpuversion0001
#BSUB -o cpuversion0001%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -cpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32G]"
#BSUB -W 24:00
#BSUB -u s173934@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/dataloader.py
echo "Done"