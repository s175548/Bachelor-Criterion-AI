from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

if __name__ == '__main__':
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = r'C:\Users\Mads-_uop20qq\Desktop\GanDataPytorchTutorial'

    # Number of workers for dataloader
    workers = 4

    # Batch size during training
    batch_size = 32 #128

    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256 # 64

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Size of z latent vector (i.e. size of generator input)
    nz = 100

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 25

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1


    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    #
    # class Generator(nn.Module):
    #     def __init__(self, class_number):
    #         super().__init__()
    #         self.linear = nn.Sequential(nn.Linear(10 * 10, 768 * 16 * 16), nn.ReLU(inplace=True))
    #         # reshape
    #         self.deconv1 = nn.Sequential(nn.ConvTranspose2d(768, 384, 3, 2, 1, 1),
    #                                      nn.BatchNorm2d(384), nn.ReLU(inplace=True))  # 32*32
    #         self.deconv2 = nn.Sequential(nn.ConvTranspose2d(384, 256, 3, 2, 1, 1),
    #                                      nn.BatchNorm2d(256), nn.ReLU(inplace=True))  # 64*64
    #         self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256, 192, 3, 2, 1, 1),
    #                                      nn.BatchNorm2d(192), nn.ReLU(inplace=True))  # 128*128
    #         # last layer no relu
    #         self.deconv4 = nn.Sequential(nn.ConvTranspose2d(192, class_number, 3, 2, 1, 1), nn.Tanh())  # 256*256
    #
    #     def forward(self, x):
    #         x = self.linear(x)
    #         x = x.reshape([-1, 768, 16, 16])
    #         x = self.deconv1(x)
    #         x = self.deconv2(x)
    #         x = self.deconv3(x)
    #         x = self.deconv4(x)
    #
    #         return x
    class Generator(nn.Module): #version
        def __init__(self, ngpu):
            super(Generator, self).__init__()
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
            return self.main(input)

    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)

    # Print the model
    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.C1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False) #in channels, #out channels, # kernelsize=4, #stride, #padding
            self.L1 = nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf) x 32 x 32
            self.C2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            self.B2 = nn.BatchNorm2d(ndf * 2)
            self.L2 = nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf*2) x 16 x 16
            self.C3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
            self.B3 = nn.BatchNorm2d(ndf * 4)
            self.L3 = nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf*4) x 8 x 8
            self.C4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
            self.B4 = nn.BatchNorm2d(ndf * 8)
            self.L4 = nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf*8) x 4 x 4
            self.C5 = nn.Conv2d(ndf * 8, 1, 4, 4, 0, bias=False)
            self.B5 = nn.BatchNorm2d(1)
            self.L5 = nn.LeakyReLU(0.2, inplace=True)

            self.C6 = nn.Conv2d(1, 1, 4, 2, 0, bias=False)
            # self.C5 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            self.S6 = nn.Sigmoid()

            # self.main = nn.Sequential(
            #     # input is (nc) x 64 x 64
            #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), #in channels, #out channels, # kernelsize=4, #stride, #padding
            #     nn.LeakyReLU(0.2, inplace=True),
            #     # state size. (ndf) x 32 x 32
            #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #     nn.BatchNorm2d(ndf * 2),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     # state size. (ndf*2) x 16 x 16
            #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #     nn.BatchNorm2d(ndf * 4),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     # state size. (ndf*4) x 8 x 8
            #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #     nn.BatchNorm2d(ndf * 8),
            #     nn.LeakyReLU(0.2, inplace=True),
            #     # state size. (ndf*8) x 4 x 4
            #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #     nn.Sigmoid()
            # )

        def forward(self, input):
            x = self.C1(input)
            x = self.L1(x)
            x = self.C2(x)
            x = self.B2(x)
            x = self.L2(x)
            x = self.C3(x)
            x = self.B3(x)
            x = self.L3(x)
            x = self.C4(x)
            x = self.B4(x)
            x = self.L4(x)
            x = self.C5(x)
            x = self.B5(x)
            x = self.L5(x)
            x = self.C6(x)
            x = self.S6(x)
            return x
            # return self.main(input)

    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)


    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    #%%capture
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)



    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()