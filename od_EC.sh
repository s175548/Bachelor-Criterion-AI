#!/bin/sh
#BSUB -J od_EC_vda
#BSUB -o od_EC_vda_%J.out
#BSUB -q gpuv100
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=128G]"
#BSUB -R "select[gpu32gb]"
#BSUB -W 0:45
#BSUB -u s175549@win.dtu.dk
#BSUB -N
# end of BSUB options

echo "Running script..."
cd ..
source test-env/bin/activate
cd Bachelor-Criterion-AI
python3 object_detect/tif_prediction.py all_bin vda no_pad original
echo "Done"