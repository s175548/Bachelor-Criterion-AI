#!/bin/sh
#BSUB -J versionSemiSupervised_false
#BSUB -o versionSemiSupervised_false%J.out
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
python3 semantic_segmentation/semi_supervised/main_setup_identical_to_supervised.py 0.01 DeepLab SGD True semi_setup_minus_semi semi_super/no_semi True False
echo "Done"