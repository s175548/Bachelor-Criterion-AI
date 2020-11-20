#!/bin/sh
#BSUB -J predictions_tif
#BSUB -o predictions_tif%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=64G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 01:00
#BSUB -u s175548@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source /zhome/db/f/128823/Bachelor/test-env/bin/activate
cd /zhome/db/f/128823/Bachelor/Bachelor-Criterion-AI
python3 semantic_segmentation/DeepLabV3/performance_metric_tif.py
echo "Done"