#!/bin/sh
#BSUB -J versionSemiSupervised01
#BSUB -o versionSemiSupervised01%J.out
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
python3 semantic_segmentation/semi_supervised/exp_gen_semi.py 0.001 DeepLab SGD False semi_setup semi_super/no_semi True False
echo "Done"