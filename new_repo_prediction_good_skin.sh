#!/bin/sh
#BSUB -J new_repo_predictions_good_skin
#BSUB -o new_repo_predictions_good_skin%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=128G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 24:00
#BSUB -u s175548@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source /zhome/db/f/128823/Bachelor/test-env/bin/activate
cd /zhome/db/f/128823/Bachelor/new_repo/Bachelor-Criterion-AI
python3 data_import/tif_evaluate_1.py
echo "Done"