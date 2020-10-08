dos2unix lr_0.0001_DeepLab.sh
dos2unix lr_0.001_DeepLab.sh
dos2unix lr_0.01_DeepLab.sh
dos2unix lr_0.0001_MobileNet.sh
dos2unix lr_0.001_MobileNet.sh
dos2unix lr_0.01_MobileNet.sh

bsub < lr_0.0001_DeepLab.sh
bsub < lr_0.001_DeepLab.sh
bsub < lr_0.01_DeepLab.sh
bsub < lr_0.0001_MobileNet.sh
bsub < lr_0.001_MobileNet.sh
bsub < lr_0.01_MobileNet.sh
