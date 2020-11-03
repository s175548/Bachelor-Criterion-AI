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
def main(semi_supervised = True):
    step_size = 1
    batch_size = 4 #
    val_batch_size = 2
    class_number=3
    lr_g=2e-4
    lr_d=0.01
    weight_decay=5e-4
    epoch_max = 130
    binary = True
    HPC = True
    # Setup random seed
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    ########### Retrive paths ##########
    if HPC:
        path_original_data, path_meta_data, save_path,path_model,dataset_path_train,dataset_path_val,datset_path_ul,model_name,exp_descrip, semi_supervised,lr_d = get_paths(binary,HPC,False,False)
        lr_g = lr_d
    else:
        path_original_data, path_meta_data, save_path,path_model,dataset_path_train,dataset_path_val,datset_path_ul= get_paths(binary,HPC,False,False)
        model_name = 'DeepLab'
        exp_descrip = ''
    model_g_spath=os.path.join(save_path,r'model_g.pt')

    ################### dataset loader ###################
    trainloader, val_loader, train_dst, _, color_dict, target_dict, annotations_dict = get_data_loaders(binary,path_original_data,path_meta_data,dataset_path_train,dataset_path_val,batch_size,val_batch_size)
    trainloader_nl, _ = get_data_loaders_unlabelled(binary,path_original_data,path_meta_data,datset_path_ul,batch_size)
    del _

    ################### build model ###################
    model_g = generator(class_number)
    model_d_dict = discriminator(model=model_name,n_classes=class_number+1) #number of classes plus an additional fake class
    model_d_dict[model_name].to(torch.device('cuda'))

    #### fine-tune ####
    #new_params=model.state_dict()
    #pretrain_dict=torch.load(r'**/model.pth')
    #pretrain_dict={k:v for k,v in pretrain_dict.items() if k in new_params and v.size()==new_params[k].size()}# default k in m m.keys
    #new_params.update(pretrain_dict)
    #model.load_state_dict(new_params)

    model_g.train()
    model_g.cuda()

    model_d_dict[model_name].train()
    model_d_dict[model_name].cuda()

    ################### optimizer ###################
    trainloader_iter = enumerate(trainloader)
    trainloader_nl_iter = enumerate(trainloader_nl)
    optimizer_g= torch.optim.Adam(model_g.parameters(),lr=lr_g,betas=(0.9,0.99),weight_decay=weight_decay)

    optim = 'Adam'
    optimizer_d = choose_optimizer(lr_d, model_name, model_d_dict, optim)
    criterion = nn.CrossEntropyLoss(ignore_index=target_dict[annotations_dict['Background']], reduction='mean')

    train_img = []
    for i in range(10,15):
        train_img.append(train_dst.__getitem__(i))

    # Set up metrics
    metrics = StreamSegMetrics(class_number+1) #3 classes plus an additional one for fake data
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=step_size, gamma=0.95)
    scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=step_size, gamma=0.95)

    train_loss_values = []
    validation_loss_values = []
    generator_losses = []
    best_score = 0
    best_scores = [0, 0, 0, 0, 0]
    best_classIoU = [0, 0, 0, 0, 0]

    ################### iter train ###################
    gamma_one = gamma_two = 0.4 #Loss weights
    epoch = 1
    iter = 0
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_loader)))
    print("starting training loop")
    while epoch <= epoch_max:
        loss_g_v=0
        loss_d_v=0
        running_loss = 0

        ####### train D ##################
        optimizer_d.zero_grad()
        # adjust_lr(optimizer_d,lr_d,iters,max_iter,power)

        # labeled data
        try:
            _,batch=next(trainloader_iter)
            iter +=1
            if iter%100==1:
                print("Epoch",epoch," and iter: ", iter,"/",len(train_dst)/val_batch_size)
        except:
            epoch += 1
            iter = 1
            print("Epoch",epoch)
            trainloader_iter=enumerate(trainloader)
            _,batch=next(trainloader_iter)


        images,labels=batch
        images=Variable(images).cuda()
        labels=Variable(labels).cuda()

        # unlabeled data
        if semi_supervised:
            try:
                _,batch_nl=next(trainloader_nl_iter)
            except:
                trainloader_nl_iter=enumerate(trainloader_nl)
                _,batch_nl=next(trainloader_nl_iter)
                del _

            images_nl=batch_nl
            images_nl = images_nl[0]
            images_nl=Variable(images_nl).cuda()
            if images.shape[0] != images_nl.shape[0]:
                continue
        # noise data
            noise = torch.rand([images.shape[0],50*50]).uniform_().cuda()
        # predict
        if model_name == 'DeepLab':
            pred_labeled = model_d_dict[model_name](images.float())['out']
        else:
            pred_labeled = model_d_dict[model_name](images.float())

        if semi_supervised:
            if model_name == 'DeepLab':
                pred_unlabel = model_d_dict[model_name](images_nl.float())['out']
                pred_fake    = model_d_dict[model_name]( model_g(noise) )['out']
            else:
                pred_unlabel = model_d_dict[model_name](images_nl.float())
                pred_fake    = model_d_dict[model_name]( model_g(noise) )
        # compute loss
#        loss_labeled = Loss_label(pred_labeled,labels)
        loss_labeled = criterion(pred_labeled, torch.tensor(labels, dtype=torch.long,device='cuda'))
        if semi_supervised:
            loss_unlabel = Loss_unlabel(pred_unlabel)
            loss_fake    = Loss_fake(pred_fake)

        if semi_supervised:
            loss_d       = loss_labeled + gamma_one*loss_fake + gamma_two*loss_unlabel
        else:
            loss_d = loss_labeled
        loss_d_v += loss_d.data.cpu().numpy().item()
        running_loss = + loss_d.item() * images.size(0)
        loss_d.backward()
        optimizer_d.step()
        if epoch % 10 == 0:
            scheduler_d.step()
            for param_group in optimizer_d.param_groups:
                print("Discriminator lr has been decreased to: ", optimizer_d.param_groups[1]['lr'])

        ####### train G ##################
        if semi_supervised:
            optimizer_g.zero_grad()
            #adjust_lr(optimizer_g,lr_g,iters,max_iter,power)

            # predict
            if model_name=='DeepLab':
                pred_fake    = model_d_dict[model_name]( model_g(noise) )['out']
            else:
                pred_fake    = model_d_dict[model_name]( model_g(noise) )
            loss_g    = -Loss_fake(pred_fake)
            loss_g_v += loss_g.data.cpu().numpy().item()
            loss_g.backward()
            optimizer_g.step()
            gen_loss = + loss_g.item() * model_g(noise).size(0)

            if epoch%10==0:
                scheduler_g.step()
                for param_group in optimizer_g.param_groups:
                    print("Generator lr has been decreased to: ",param_group['lr'])

        # output loss value, and validate
        if (epoch%1 == 0 and iter==1):
            print('epoch={} , loss_g={} , loss_d={}'.format(epoch,loss_g_v,loss_d_v))
            print("validation...")
            model_d_dict[model_name].eval()
            val_score, ret_samples, validation_loss = validate(ret_samples_ids=range(5),
                                                               model=model_d_dict[model_name], loader=val_loader, device='cuda',
                                                               metrics=metrics, model_name=model_name, N=epoch,
                                                               criterion=criterion, train_images=train_img, lr=optimizer_d.param_groups[1]['lr'],
                                                               save_path=save_path,
                                                               color_dict=color_dict, target_dict=target_dict,
                                                               annotations_dict=annotations_dict,
                                                               exp_description='semi_super')
            print(metrics.to_str(val_score))

            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                best_scores.append(best_score)
                best_classIoU.append([val_score['Class IoU']])
                best_classIoU = [x for _, x in sorted(zip(best_scores, best_classIoU), reverse=True)][:5]
                best_scores.sort(reverse=True)
                best_scores = best_scores[:5]
                save_ckpt(model=model_d_dict[model_name], model_name=model_name, cur_itrs=iter, optimizer=optimizer_d,
                          scheduler=scheduler_d, best_score=best_score, lr=optimizer_d.param_groups[1]['lr'], save_path=save_path,exp_description=exp_descrip)
                torch.save(model_g.state_dict(), model_g_spath)
                np.save('metrics', metrics.to_str(val_score))
                print("[Val] Overall Acc", iter, val_score['Overall Acc'])
                print("[Val] Mean IoU", iter, val_score['Mean IoU'])
                print("[Val] Class IoU", val_score['Class IoU'])
            elif val_score['Mean IoU'] > min(best_scores):
                best_scores.append(val_score['Mean IoU'])
                best_classIoU.append(val_score['Class IoU'])
                best_classIoU = [x for _, x in sorted(zip(best_scores, best_classIoU), reverse=True)][:5]
                best_scores.sort(reverse=True)
                best_scores = best_scores[:5]
            model_d_dict[model_name].train()

            validation_loss_values.append(validation_loss /len(val_loader))
            train_loss_values.append(running_loss / len(train_dst))
            generator_losses.append(-gen_loss)

    plt.plot(range(len(generator_losses)), generator_losses, '-o')
    plt.title('Train Loss')
    plt.xlabel('N_epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_path, exp_descrip + (str(lr_d)) + '_generator_loss'), format='png')
    plt.close()
    save_plots_and_parameters(best_classIoU, best_scores, True, exp_descrip, optimizer_d.param_groups[1]['lr'], metrics, model_d_dict[model_name],
                              model_name, optim, save_path, train_loss_values, val_score, validation_loss_values)


if __name__ == '__main__':
    main(semi_supervised=True)


