#!/bin/sh
#BSUB -J versionSemiSupervised01_softmax_onelayer
#BSUB -o versionSemiSupervised01_softmax_onelayer%J.out
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
cd /zhome/87/9/127623/BachelorProject/cropped_data/semi/Bachelor-Criterion-AI
python3 semantic_segmentation/semi_supervised/exp_gen_semi.py 0.006 DeepLab SGD False softmax_onelayer semi_super/softmax_only_one/D006G004 True True False
echo "Done"