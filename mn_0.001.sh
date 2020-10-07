#!/bin/sh
#BSUB -J mobilenet
#BSUB -o mobilenet%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=128G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -u s175549@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 object_detect/train_hpc.py 0.001 mobilenet
echo "Done"