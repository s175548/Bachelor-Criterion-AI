'''
  the author is leilei
'''
import sys,os
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')

import torch
from torchvision import transforms
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils import data
# imread data
#from data_imread import batch_data
# models
from semantic_segmentation.semi_supervised.generator import generator
from semantic_segmentation.semi_supervised.discriminator import discriminator
# losses
from semantic_segmentation.semi_supervised.losses import Loss_label, Loss_fake, Loss_unlabel

from semantic_segmentation.DeepLabV3.Training_windows import *
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData
from data_import.data_loader import DataLoader
import argparse,json,ast
from PIL import Image
from torchvision import transforms
from semantic_segmentation.DeepLabV3.network.utils import randomCrop,pad
from semantic_segmentation.DeepLabV3.Training_windows import my_def_collate
from semantic_segmentation.semi_supervised.helpful_functions import get_paths, get_data_loaders,get_data_loaders_unlabelled
from semantic_segmentation.DeepLabV3.Training_windows import validate
################### Hyper parameter ################### 
def main():
    #batch_size=16
    batch_size = 16 #
    val_batch_size = 4
    class_number=3
    lr_g=2e-4
    lr_d=1e-4
    power=0.9
    weight_decay=5e-4
    max_iter=20000
    binary = True
    HPC = False

    # Setup random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    path_original_data, path_meta_data, save_path,path_model,dataset_path_train,dataset_path_val,datset_path_ul = get_paths(binary,HPC,False,False)
    model_s_path=os.path.join(save_path,r'model.pt')
    model_g_spath=os.path.join(save_path,r'model_g.pt')
    ################### update lr ###################
    def lr_poly(base_lr,iters,max_iter,power):
        return base_lr*((1-float(iters)/max_iter)**power)
    def adjust_lr(optimizer,base_lr,iters,max_iter,power):
        lr=lr_poly(base_lr,iters,max_iter,power)
        optimizer.param_groups[0]['lr']=lr
        if len(optimizer.param_groups)>1:
            optimizer.param_groups[1]['lr']=lr*10

    ################### dataset loader ###################
    trainloader, val_loader, train_dst, _, color_dict, target_dict, annotations_dict = get_data_loaders(binary,path_original_data,path_meta_data,dataset_path_train,dataset_path_val,batch_size,val_batch_size)
    trainloader_nl, _ = get_data_loaders_unlabelled(binary,path_original_data,path_meta_data,datset_path_ul,batch_size)


    ################### build model ###################
    model_name = 'MobileNet'
    model_g = generator(class_number)
    model_d_dict = discriminator(model=model_name,n_classes=class_number)
    model_d = model_d_dict[model_name]
    model_d.to(torch.device('cuda'))

    #### fine-tune ####
    #new_params=model.state_dict()
    #pretrain_dict=torch.load(r'**/model.pth')
    #pretrain_dict={k:v for k,v in pretrain_dict.items() if k in new_params and v.size()==new_params[k].size()}# default k in m m.keys
    #new_params.update(pretrain_dict)
    #model.load_state_dict(new_params)

    model_g.train()
    model_g.cuda()

    model_d.train()
    model_d.cuda()

    ################### optimizer ###################
    trainloader_iter = enumerate(trainloader)
    trainloader_nl_iter = enumerate(trainloader_nl)
    optimizer_g=torch.optim.Adam(model_g.parameters(),lr=lr_g,betas=(0.9,0.99),weight_decay=weight_decay)
    #optimizer_g.zero_grad()

    optimizer_d=torch.optim.Adam(model_d.parameters(),lr=lr_d,betas=(0.9,0.99),weight_decay=weight_decay)
    #optimizer_d.zero_grad()

    train_img = []
    for i in range(10,12):
        train_img.append(train_dst.__getitem__(i))

    # Set up metrics
    metrics = StreamSegMetrics(class_number)
    ################### iter train ###################
    epoch = 1
    for iters in range(max_iter):
        loss_g_v=0
        loss_d_v=0
        running_loss = 0


        ####### train D ##################
        optimizer_d.zero_grad()
        adjust_lr(optimizer_d,lr_d,iters,max_iter,power)

        # labeled data
        try:
            _,batch=next(trainloader_iter)
        except:
            print("Epoch",epoch)
            epoch+=1
            trainloader_iter=enumerate(trainloader)
            _,batch=next(trainloader_iter)

        images,labels=batch
        images=Variable(images).cuda()
        labels=Variable(labels).cuda()

        # unlabeled data
        try:
            _,batch_nl=next(trainloader_nl_iter)
        except:
            trainloader_nl_iter=enumerate(trainloader_nl)
            _,batch_nl=next(trainloader_nl_iter)

        images_nl=batch_nl
        images_nl = images_nl[0]
        images_nl=Variable(images_nl).cuda()
        if images.shape[0] != images_nl.shape[0]:
            continue
        # noise data
        noise = torch.rand([images.shape[0],50*50]).uniform_().cuda()
        # predict
        pred_labeled = model_d(images.float())
        pred_unlabel = model_d(images_nl.float())
        pred_fake    = model_d( model_g(noise) )
        # compute loss
#        loss_labeled = Loss_label(pred_labeled,labels)
        criterion = nn.CrossEntropyLoss(ignore_index=class_number+2, reduction='mean')
        loss_labeled = criterion(pred_labeled, torch.tensor(labels, dtype=torch.long,device='cuda'))
        loss_unlabel = Loss_unlabel(pred_unlabel)
        loss_fake    = Loss_fake(pred_fake)

        gamma_one = gamma_two = 0.4
        loss_d       = loss_labeled + gamma_one*loss_fake + gamma_two*loss_unlabel
        loss_d_v += loss_d.data.cpu().numpy().item()
        loss_d.backward()
        optimizer_d.step()

        ####### train G ##################
        optimizer_g.zero_grad()
        adjust_lr(optimizer_g,lr_g,iters,max_iter,power)
        # predict
        pred_fake    = model_d( model_g(noise) )
        loss_g    = -Loss_fake(pred_fake)
        loss_g_v += loss_g.data.cpu().numpy().item()
        loss_g.backward()
        optimizer_g.step()

        # output loss value, and validate
        if (iters%200 == 0 and iters != 0):
            print('iter={} , loss_g={} , loss_d={}'.format(iters,loss_g_v,loss_d_v))
            print("validation...")
            model_d.eval()
            val_score, ret_samples, validation_loss = validate(ret_samples_ids=range(2),
                                                               model=model_d, loader=val_loader, device='cuda',
                                                               metrics=metrics, model_name=model_name, N=iters,
                                                               criterion=criterion, train_images=train_img, lr=0.1,
                                                               save_path=save_path,
                                                               color_dict=color_dict, target_dict=target_dict,
                                                               annotations_dict=annotations_dict,
                                                               exp_description='semi_super')
            print(metrics.to_str(val_score))
        model_d.train()

        # save model
        if iters%25==0:
            print(iters)
            if iters%200==0:
                # fake_dat = model_g(noise)[0].cpu().detach().numpy()
                # fake_img = Image.fromarray((fake_dat * 255).reshape(256, 256, -1).astype('uint8'))
                # Image._show(fake_img)
                # test image

                torch.save(model_d.state_dict(),model_s_path)
                torch.save(model_g.state_dict(),model_g_spath)
    torch.save(model_d.state_dict(),model_s_path)
    torch.save(model_g.state_dict(),model_g_spath)


if __name__ == '__main__':
    main()


