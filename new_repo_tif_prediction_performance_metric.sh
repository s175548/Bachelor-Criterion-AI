#!/bin/sh
#BSUB -J new_repo_perform_tif
#BSUB -o new_repo_perform_tif%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 00:15
#BSUB -u s175548@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source /zhome/db/f/128823/Bachelor/test-env/bin/activate
cd /zhome/db/f/128823/Bachelor/new_repo/Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/performance_metric_tif.py
echo "Done"