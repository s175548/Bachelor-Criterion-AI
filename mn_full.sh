#!/bin/sh
#BSUB -J mobilenet_full_
#BSUB -o mobilenet_full_%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -u s175549@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 object_detect/train_hpc.py 0.01 mobilenet SGD full
echo "Done"