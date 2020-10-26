"""
Made with heavy inspiration from
https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/af50e37932732a2c06e331c54cc8c64820c307f4/main.py
"""
import sys
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')


from tqdm import tqdm
import random,json
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
from semantic_segmentation.DeepLabV3.utils.utils import Denormalize
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import os
import PIL
import pickle
import matplotlib.pyplot as plt
from data_import.data_loader import convert_to_image
from semantic_segmentation.DeepLabV3.network.modeling import _segm_mobilenet

num_classes=2
output_stride=16
save_val_results=False
total_itrs=1000#1000
#lr=0.01 # Is a parameter in training()
lr_policy='step'
step_size=10000
batch_size= 2 # 16
val_batch_size= 4 #4
loss_type="cross_entropy"
weight_decay=1e-4
random_seed=1
val_interval= 55 # 55
vis_num_samples= 2 #2
enable_vis=True
N_epochs= 100





def save_ckpt(model,model_name=None,cur_itrs=None, optimizer=None,scheduler=None,best_score=None,save_path = os.getcwd(),lr=0.01,exp_description=''):
    """ save current model
    """
    torch.save({"cur_itrs": cur_itrs,"model_state": model.state_dict(),"optimizer_state": optimizer.state_dict(),"scheduler_state": scheduler.state_dict(),"best_score": best_score,
    }, os.path.join(save_path,"{}_{}".format(model_name,exp_description)+str(lr)+'.pt'))
    print("Model saved as "+model_name+'.pt')

def validate(model,model_name, loader, device, metrics,N,criterion,
             ret_samples_ids=None,save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_predictions/',
             train_images=None,lr=0.01,color_dict=None,target_dict=None,annotations_dict=None,exp_description=''):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    running_loss=0
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if model_name=='DeepLab':
                outputs = model(images)['out']
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss = + loss.item() * images.size(0)


            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0], targets[0], preds[0]))

            metrics.update(targets, preds)


        if N%10 == 0 or N==1:
            for (image,target,pred), id in zip(ret_samples,ret_samples_ids):
                target = convert_to_image(target.squeeze(), color_dict, target_dict)
                pred = convert_to_image(pred.squeeze(), color_dict, target_dict)
                image = (denorm(image.detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                PIL.Image.fromarray(image.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_img.png'.format(model_name,N,id,"{}".format(exp_description)+str(lr) )),format='PNG' )
                PIL.Image.fromarray(pred.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_prediction.png'.format(model_name,N,id,"{}".format(exp_description)+str(lr) )),format='PNG')
                PIL.Image.fromarray(target.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_target.png'.format(model_name,N,id,"{}".format(exp_description)+str(lr) )),format='PNG')


            for i in range(len(train_images)):
                image = train_images[i][0].unsqueeze(0)
                image = image.to(device, dtype=torch.float32)
                if model_name=='DeepLab':
                    output = model(image)['out']
                else:
                    output = model(image)

                pred = output.detach().max(dim=1)[1].cpu().squeeze().numpy()
                target=train_images[i][1].cpu().squeeze().numpy()
                target=convert_to_image(target.squeeze(),color_dict,target_dict)
                pred=convert_to_image(pred.squeeze(),color_dict,target_dict)
                image = (denorm(train_images[i][0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
                PIL.Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path,'{}_{}_{}_{}_img_train.png'.format(model_name, N, i,"{}".format(exp_description)+str(lr))),
                                                                 format='PNG')
                PIL.Image.fromarray(pred.astype(np.uint8)).save(os.path.join(save_path , '{}_{}_{}_{}_prediction_train.png'.format(model_name, N, i, "{}".format(exp_description) + str(lr)) ), format='PNG')
                PIL.Image.fromarray(target.astype(np.uint8)).save(os.path.join( save_path , '{}_{}_{}_{}_mask_train.png'.format(model_name, N, i,"{}".format(exp_description)+str(lr)) ), format='PNG')


        score = metrics.get_results()
        print(score)
    return score, ret_samples,running_loss



def training(n_classes=3,model='DeepLab',load_models=False,model_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /',
             train_loader=None,val_loader=None,train_dst=None, val_dst=None,
             save_path = os.getcwd(),lr=0.01,train_images = None,color_dict=None,target_dict=None,annotations_dict=None,exp_description = '',optim='SGD',default_scope = True):


    model_dict={}
    if model=='DeepLab':
        model_dict[model]=deeplabv3_resnet101(pretrained=True, progress=True,num_classes=21, aux_loss=None)
        if default_scope:
            grad_check(model_dict[model])
        else:
            grad_check(model_dict[model], model_layers='All')
        model_dict[model].classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
        model_dict[model].aux_classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()


    if model=="MobileNet":
        model_dict[model] = _segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=n_classes+2,pretrained_backbone=True)
        if default_scope:
            grad_check(model_dict[model],model_layers='All')
        else:
            grad_check(model_dict[model])



    # Setup visualization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


    # Set up metrics
    metrics = StreamSegMetrics(n_classes+2)

    # Set up optimizer
    if optim == 'SGD':
        optimizer = torch.optim.SGD(params=[
            {'params': model_dict[model].backbone.parameters(), 'lr': 0.3 * lr},
            {'params': model_dict[model].classifier.parameters(), 'lr': lr},
        ], lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'Adam':
        optimizer = torch.optim.Adam(params=[
            {'params': model_dict[model].backbone.parameters(), 'lr': 0.3 * lr},
            {'params': model_dict[model].classifier.parameters(), 'lr': lr},
        ], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.RMSprop(params=[
            {'params': model_dict[model].backbone.parameters(), 'lr': 0.3 * lr},
            {'params': model_dict[model].classifier.parameters(), 'lr': lr},
        ], lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=n_classes+2, reduction='mean')


    # ==========   Train Loop   ==========#


    for model_name, model in model_dict.items():
        print(model_name)
        cur_epochs = 0
        interval_loss = 0
        train_loss_values = []
        validation_loss_values=[]
        best_score = 0
        best_scores = [0,0,0,0,0]
        best_classIoU = [0,0,0,0,0]
        model.to(device)
        while cur_epochs<N_epochs :  # cur_itrs < opts.total_itrs:
            model.train()
            cur_itrs=0
            cur_epochs += 1
            running_loss = 0
            for images, labels in tqdm(train_loader):
                cur_itrs += 1

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                optimizer.zero_grad()
                if model_name=='DeepLab':
                    outputs = model(images)['out']
                else:
                    outputs = model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                running_loss = + loss.item() * images.size(0)
                interval_loss += np_loss
                print('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 1 == 0:
                    interval_loss = interval_loss / images.size(0)
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, total_itrs, interval_loss))
                    interval_loss = 0.0

                # if (cur_itrs) % np.floor(len(train_dst)/batch_size) == 0:
                if True:
                    print("validation...")
                    model.eval()
                    val_score, ret_samples,validation_loss = validate(ret_samples_ids=range(5),
                        model=model, loader=val_loader, device=device, metrics=metrics,model_name=model_name,N=cur_epochs,criterion=criterion,train_images=train_images,lr=lr,save_path=save_path,
                                                                      color_dict=color_dict,target_dict=target_dict,annotations_dict=annotations_dict,exp_description=exp_description)
                    print(metrics.to_str(val_score))
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        best_scores.append(best_score)
                        best_classIoU.append([val_score['Class IoU']])
                        best_classIoU = [x for _, x in sorted(zip(best_scores, best_classIoU),reverse=True)][:5]
                        best_scores.sort(reverse=True)
                        best_scores = best_scores[:5]
                        save_ckpt(model=model,model_name=model_name,cur_itrs=cur_itrs, optimizer=optimizer, scheduler=scheduler, best_score=best_score,lr=lr,save_path=save_path,exp_description=exp_description)
                        np.save('metrics',metrics.to_str(val_score))
                        print("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        print("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        print("[Val] Class IoU", val_score['Class IoU'])
                    elif val_score['Mean IoU'] > min(best_scores):
                        best_scores.append(val_score['Mean IoU'])
                        best_classIoU.append(val_score['Class IoU'])
                        best_classIoU = [x for _, x in sorted(zip(best_scores, best_classIoU),reverse=True)][:5]
                        best_scores.sort(reverse=True)
                        best_scores = best_scores[:5]
                    model.train()
                scheduler.step()

                if cur_itrs >= total_itrs:
                    break
            validation_loss_values.append(validation_loss /len(val_dst))
            train_loss_values.append(running_loss / len(train_dst))

        plt.plot(range(N_epochs),train_loss_values,'-o')
        plt.title('Train Loss')
        plt.xlabel('N_epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path,exp_description+(str(lr))+'_train_loss'),format='png')
        plt.close()
        plt.plot(range(N_epochs),validation_loss_values, '-o')
        plt.title('Validation Loss')
        plt.xlabel('N_epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path, exp_description + (str(lr)) + '_val_loss'), format='png')
        plt.close()

        experiment_dict = {}
        best_metric = metrics.to_str(val_score)
        hyperparams_val = [N_epochs,lr,batch_size,val_batch_size,loss_type,weight_decay,optim,random_seed,best_metric,best_scores,best_classIoU,model_name,default_scope,model]
        hyperparams = ['N_epochs','lr','batch_size','val_batch_size','loss_type','weight_decay','optimizer','random_seed','best_metric','best_scores','best_classIoU','model_backbone','default_scope','model architecture']
        for idx,key in enumerate(hyperparams):
            experiment_dict[key] = hyperparams_val[idx]
        with open("{}/{}_{}.txt".format(save_path,model_name,exp_description), "w") as text_file:
            text_file.write(str(experiment_dict))





def grad_check(model,model_layers='Classifier'):
    if model_layers=='Classifier':
        print('Classifier only')
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(requires_grad=True)
    else:
        print('Whole model')
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad=True)

import random,cv2
def randomCrop(img, mask, width, height):
    crop_img = np.empty((3,width,height))
    crop_mask = np.empty((width,height))
    x = random.randint(0, img.shape[1]-width)
    y = random.randint(0,img.shape[2]- height)
    img_num = img.numpy()
    mask_num = mask.numpy()
    for i in range(3):
        crop_img[i] = img_num[i][x:x+width,y:y+height ]
    crop_mask = mask_num[x:x+width,y:y+height ]
    return torch.from_numpy(crop_img),torch.from_numpy(crop_mask)
def pad(img,mask,size,ignore_idx):
    #        topBorderWidth,bottomBorderWidth, leftBorderWidth,  rightBorderWidth,
    img_num = img.numpy()
    mask_num = mask.numpy()
    pad_img = np.empty((3,size,size))
    pad_mask = np.empty(( size, size))

    height_border = (size-img_num.shape[1] )// 2
    width_border = (size - img_num.shape[2]) //2
    if (size-img_num.shape[1])%2 != 0:
        rest_height = 1
    else:
        rest_height = 0
    if (size-img_num.shape[2])%2 != 0:
        rest_width = 1
    else:
        rest_width = 0

    if (width_border >= 0 and height_border >= 0):
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], height_border, height_border + rest_height, width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)
        pad_mask = cv2.copyMakeBorder(mask_num, height_border, height_border + rest_height, width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)
    elif height_border >= 0:
        rand_start = random.randint(0, abs(width_border) * 2)
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], height_border, height_border + rest_height, 0,0, cv2.BORDER_CONSTANT, value=ignore_idx)[:,rand_start:rand_start+size]
        pad_mask = cv2.copyMakeBorder(mask_num, height_border, height_border + rest_height, 0,0, cv2.BORDER_CONSTANT, value=ignore_idx)[:,rand_start:rand_start+size]
    elif width_border >= 0:
        rand_start = random.randint(0,abs(height_border)*2)
        for i in range(3):
            pad_img[i] = cv2.copyMakeBorder(img_num[i], 0, 0 , width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)[rand_start:rand_start+size,:]
        pad_mask = cv2.copyMakeBorder(mask_num, 0, 0 , width_border,width_border + rest_width, cv2.BORDER_CONSTANT, value=ignore_idx)[rand_start:rand_start+size,:]

    return torch.from_numpy(pad_img),torch.from_numpy(pad_mask)

def my_def_collate(batch,size=1200):
    IGNORE_INDEX = 4
    for idx,item in enumerate(batch):
        # transform_function = et.ExtCompose([et.ExtRandomHorizontalFlip(p=0.5), et.ExtRandomVerticalFlip(p=0.5), et.ExtEnhanceContrast(),et.ExtToTensor(), et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # pil_image = transforms.ToPILImage()(item[0]).convert("RGB")
        # pil_label = transforms.ToPILImage()(item[1])
        if (item[0].shape[1] >= size and item[0].shape[2]>= size):
            img, mask =randomCrop(item[0],item[1],size,size)
        else:
            img, mask = pad(item[0],item[1],size,IGNORE_INDEX)
        if idx ==0:
            data = img
            masks = mask
        elif idx==1:
            data = torch.stack([data,img])
            masks = torch.stack([masks,mask])
        else:
            data = torch.cat([data,img.unsqueeze(0)])
            masks = torch.cat([masks,mask.unsqueeze(0)])
    return data,masks

if __name__ == "__main__":

    #training(['model_pre_full'],path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg')
    pass
