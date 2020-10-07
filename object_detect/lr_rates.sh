dos2unix mn_0.01.sh
dos2unix mn_0.001.sh
dos2unix rn_0.01.sh
dos2unix rn_0.001.sh

bsub < mn_0.01.sh
bsub < mn_0.001.sh
bsub < rn_0.01.sh
bsub < rn_0.001.sh
