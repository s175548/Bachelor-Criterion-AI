from torchfcn.models import FCN8s
from torchfcn.trainer import Trainer
from torch.optim import adam
opt= adam
FCN_8=FCN8s()
trainer= Trainer(cuda=False,model=FCN_8,optimizer=adam,
