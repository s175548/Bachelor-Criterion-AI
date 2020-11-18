'''
  the author is leilei
'''

from torch import nn

class Generator(nn.Module):
    def __init__(self,class_number):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(10*10,768*16*16),nn.ReLU(inplace=True))
        # reshape
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(768,384,3,2,1,1),
                                     nn.BatchNorm2d(384),nn.ReLU(inplace=True))#32*32
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(384,256,3,2,1,1),
                                     nn.BatchNorm2d(256),nn.ReLU(inplace=True))#64*64
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,192,3,2,1,1),
                                     nn.BatchNorm2d(192),nn.ReLU(inplace=True))#128*128
        # last layer no relu
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(192,class_number,3,2,1,1),nn.Tanh())#256*256
    def forward(self,x):
        x = self.linear(x)
        x = x.reshape([-1,768,16,16])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        return x


def generator1(class_number):
    model = Generator(class_number)
    return model

######################################################################
class Generator(nn.Module):
    def __init__(self,class_number):
        super().__init__()
        # input [N,50*50] 由于全连接层 4096*4096 就很大了，因此这里不能设置那么大
        self.linear = nn.Sequential(nn.Linear(50*50,64*16*16),nn.ReLU(inplace=True))
        # reshape
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64,128,3,2,1,1),
                                     nn.BatchNorm2d(128),nn.ReLU(inplace=True))#32*32
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128,256,3,2,1,1),
                                     nn.BatchNorm2d(256),nn.ReLU(inplace=True))#64*64
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,128,3,2,1,1),
                                     nn.BatchNorm2d(128),nn.ReLU(inplace=True))#128*128
        # last layer no relu
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(128,class_number,3,2,1,1),nn.Tanh())#256*256
        
    def forward(self,x):
        x = self.linear(x)
        x = x.reshape([-1,64,16,16])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        
        return x




class Generator_pytorchTut(nn.Module): #version
    def __init__(self, ngpu,spectral=False):
        # Number of channels in the training images. For color images this is 3
        nc = 3
        super(Generator_pytorchTut, self).__init__()
        nz = 100
        # Size of feature maps in generator
        ngf = 64
        # Size of feature maps in discriminator
        ndf = 64
        if not spectral:
            self.ngpu = ngpu
            self.C1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
            self.B1 = nn.BatchNorm2d(ngf * 8)
            self.Rel1 = nn.ReLU(True)

            self.C2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
            self.B2 = nn.BatchNorm2d(ngf * 4)
            self.Rel2 = nn.ReLU(True)
                # state size. (ngf*4) x 8 x 8
            self.C3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
            self.B3 = nn.BatchNorm2d(ngf * 2)
            self.Rel3 = nn.ReLU(True)
                # state size. (ngf*2) x 16 x 16
            self.C4 = nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)
            self.B4 = nn.BatchNorm2d(ngf)
            self.Rel4 = nn.ReLU(True)
                # state size. (ngf) x 32 x 32
            self.C5 = nn.ConvTranspose2d(ngf, int(ngf/2), 4, 2, 1, bias=False)
            self.B5 = nn.BatchNorm2d(int(ngf/2))
            self.Rel5 = nn.ReLU(True)

            self.C6 = nn.ConvTranspose2d(int(ngf/2), int(ngf/4), 4, 2, 1, bias=False)
            self.B6 = nn.BatchNorm2d(int(ngf/4))
            self.Rel6 = nn.ReLU(True)

            self.C7 = nn.ConvTranspose2d(int(ngf/4), nc, 4, 2, 1, bias=False)
            self.T7 = nn.Tanh()
            # state size. (nc) x 64 x 64
        else:
            self.ngpu = ngpu
            self.C1 = nn.utils.spectral_norm(nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False))
            self.B1 = nn.BatchNorm2d(ngf * 8)
            self.Rel1 = nn.ReLU(True)

            self.C2 = nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
            self.B2 = nn.BatchNorm2d(ngf * 4)
            self.Rel2 = nn.ReLU(True)
            # state size. (ngf*4) x 8 x 8
            self.C3 = nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
            self.B3 = nn.BatchNorm2d(ngf * 2)
            self.Rel3 = nn.ReLU(True)
            # state size. (ngf*2) x 16 x 16
            self.C4 = nn.utils.spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))
            self.B4 = nn.BatchNorm2d(ngf)
            self.Rel4 = nn.ReLU(True)
            # state size. (ngf) x 32 x 32
            self.C5 = nn.utils.spectral_norm(nn.ConvTranspose2d(ngf, int(ngf / 2), 4, 2, 1, bias=False))
            self.B5 = nn.BatchNorm2d(int(ngf / 2))
            self.Rel5 = nn.ReLU(True)

            self.C6 = nn.utils.spectral_norm(nn.ConvTranspose2d(int(ngf / 2), int(ngf / 4), 4, 2, 1, bias=False))
            self.B6 = nn.BatchNorm2d(int(ngf / 4))
            self.Rel6 = nn.ReLU(True)

            self.C7 = nn.utils.spectral_norm(nn.ConvTranspose2d(int(ngf / 4), nc, 4, 2, 1, bias=False))
            self.T7 = nn.Tanh()

        self.main = nn.Sequential(
                        # input is Z, going into a convolution
                        nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(ngf * 8),
                        nn.ReLU(True),
                        # state size. (ngf*8) x 4 x 4
                        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 4),
                        nn.ReLU(True),
                        # state size. (ngf*4) x 8 x 8
                        nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf * 2),
                        nn.ReLU(True),
                        # state size. (ngf*2) x 16 x 16
                        nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(ngf),
                        nn.ReLU(True),
                        # state size. (ngf) x 32 x 32
                        nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                        nn.Tanh()
                        # state size. (nc) x 64 x 64
                    )

    def forward(self, input):
        # return self.main(input)
        x = self.C1(input)
        x = self.B1(x)
        x = self.Rel1(x)
        x = self.C2(x)
        x = self.B2(x)
        x = self.Rel2(x)
        x = self.C3(x)
        x = self.B3(x)
        x = self.Rel3(x)
        x = self.C4(x)
        x = self.B4(x)
        x = self.Rel4(x)
        x = self.C5(x)
        x = self.B5(x)
        x = self.Rel5(x)
        x = self.C6(x)
        x = self.B6(x)
        x = self.Rel6(x)
        x = self.C7(x)
        x = self.T7(x)
        return x
        # return self.main(input)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def generator(class_number,spectral=False):
    model = Generator_pytorchTut(1,spectral)
    return model