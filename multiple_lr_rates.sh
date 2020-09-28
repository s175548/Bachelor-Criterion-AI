dos2unix lr_0.0001.sh
dos2unix lr_0.001.sh
dos2unix lr_0.01.sh


bsub < lr_0.0001.sh
bsub < lr_0.001.sh
bsub < lr_0.01.sh
