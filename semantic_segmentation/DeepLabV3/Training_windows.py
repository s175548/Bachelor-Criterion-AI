"""
Made with heavy inspiration from
https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/af50e37932732a2c06e331c54cc8c64820c307f4/main.py
"""
import sys
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI')
sys.path.append('/zhome/87/9/127623/BachelorProject/Bachelor-Criterion-AI/semantic_segmentation')


from tqdm import tqdm
import random
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









#Forskellig learning rate, (træn på en klasse, tick bite, multiclass)

num_classes=2
output_stride=16
save_val_results=False
total_itrs=1000#1000
#lr=0.01 # Is a parameter in training()
lr_policy='step'
step_size=10000
batch_size= 16 # 16
val_batch_size= 4 #4
loss_type="cross_entropy"
weight_decay=1e-4
random_seed=1
val_interval= 55 # 55
vis_num_samples= 2 #2
enable_vis=True 
N_epochs= 100 # 240 #Helst mange





def save_ckpt(model,model_name=None,cur_itrs=None, optimizer=None,scheduler=None,best_score=None,save_path = os.getcwd(),lr=0.01):
    """ save current model
    """
    torch.save({"cur_itrs": cur_itrs,"model_state": model.state_dict(),"optimizer_state": optimizer.state_dict(),"scheduler_state": scheduler.state_dict(),"best_score": best_score,
    }, os.path.join(save_path,"model_tick"+str(lr)+'.pt'))
    print("Model saved as "+model_name+'.pt')

def validate(model,model_name, loader, device, metrics,N,criterion,
             ret_samples_ids=None,save_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_predictions/',
             train_images=None,lr=0.01,color_dict=None,target_dict=None,annotations_dict=None):
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

            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            running_loss = + loss.item() * images.size(0)


            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()


            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0], targets[0], preds[0]))

            metrics.update(targets, preds)



        for (image,target,pred), id in zip(ret_samples,ret_samples_ids):
            target = convert_to_image(target.squeeze(), color_dict, target_dict)
            pred = convert_to_image(pred.squeeze(), color_dict, target_dict)
            image = (denorm(image.detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
            PIL.Image.fromarray(image.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_img.png'.format(model_name,N,id,"tick"+str(lr) )),format='PNG' )
            PIL.Image.fromarray(pred.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_prediction.png'.format(model_name,N,id,"tick"+str(lr) )),format='PNG')
            PIL.Image.fromarray(target.astype(np.uint8)).save( os.path.join( save_path,r'{}_{}_{}_{}_target.png'.format(model_name,N,id,"tick"+str(lr) )),format='PNG')




        for i in range(len(train_images)):
            image = train_images[i][0].unsqueeze(0)
            image = image.to(device, dtype=torch.float32)
            output = model(image)['out']
            pred = output.detach().max(dim=1)[1].cpu().squeeze().numpy()
            target=train_images[i][1].cpu().squeeze().numpy()
            target=convert_to_image(target.squeeze(),color_dict,target_dict)
            pred=convert_to_image(pred.squeeze(),color_dict,target_dict)
            image = (denorm(train_images[i][0].detach().cpu().numpy()) * 255).transpose(1, 2, 0).astype(np.uint8)
            PIL.Image.fromarray(image.astype(np.uint8)).save(os.path.join(save_path,'{}_{}_{}_{}_img_train.png'.format(model_name, N, i,"tick"+str(lr))),
                                                             format='PNG')
            PIL.Image.fromarray(pred.astype(np.uint8)).save(os.path.join(save_path , '{}_{}_{}_{}_prediction_train.png'.format(model_name, N, i, "tick" + str(lr)) ), format='PNG')
            PIL.Image.fromarray(target.astype(np.uint8)).save(os.path.join( save_path , '{}_{}_{}_{}_mask_train.png'.format(model_name, N, i,"tick"+str(lr)) ), format='PNG')


        score = metrics.get_results()
        print(score)
    return score, ret_samples,running_loss



def training(n_classes=3,model='Deep_lab',load_models=False,model_path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /',
             train_loader=None,val_loader=None,train_dst=None, val_dst=None,
             save_path = os.getcwd(),lr=0.01,train_images = None,color_dict=None,target_dict=None,annotations_dict=None):


    model_dict={}
    if model=='DeepLab':
        model_dict[model]=deeplabv3_resnet101(pretrained=True, progress=True,
                                  num_classes=21, aux_loss=None)
        grad_check(model_dict[model])
        model_dict[model].classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()
        model_dict[model].aux_classifier[-1] = torch.nn.Conv2d(256, n_classes+2, kernel_size=(1, 1), stride=(1, 1)).requires_grad_()


    if model=="MobileNet":
        model_dict[model] = _segm_mobilenet('deeplabv3', 'mobile_net', output_stride=8, num_classes=num_classes+1,
                                     pretrained_backbone=True)
        grad_check(model_dict[model],model_layers='All')



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
    optimizer = torch.optim.SGD(params=[
        {'params': model_dict[model].backbone.parameters(), 'lr': 0.3 * lr},
        {'params': model_dict[model].classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    criterion = nn.CrossEntropyLoss(ignore_index=n_classes+2, reduction='mean')


    # ==========   Train Loop   ==========#


    for model_name, model in model_dict.items():
        cur_epochs = 0
        interval_loss = 0
        train_loss_values = []
        validation_loss_values=[]
        best_score = 0
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
                outputs = model(images)['out']

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                running_loss = + loss.item() * images.size(0)
                interval_loss += np_loss
                print('Loss', cur_itrs, np_loss)

                if (cur_itrs) % 1 == 0:
                    interval_loss = interval_loss / 10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, total_itrs, interval_loss))
                    interval_loss = 0.0

                if (cur_itrs) % np.floor(len(train_dst)/batch_size) == 0:
                    print("validation...")
                    model.eval()
                    val_score, ret_samples,validation_loss = validate(ret_samples_ids=range(5),
                        model=model, loader=val_loader, device=device, metrics=metrics,model_name=model_name,N=cur_epochs,criterion=criterion,train_images=train_images,lr=lr,save_path=save_path,
                                                                      color_dict=color_dict,target_dict=target_dict,annotations_dict=annotations_dict)
                    print(metrics.to_str(val_score))
                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        save_ckpt(model=model,cur_itrs=cur_itrs, optimizer=optimizer, scheduler=scheduler, best_score=best_score,model_name=model_name,lr=lr,save_path=save_path)
                        print("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        print("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        print("[Val] Class IoU", val_score['Class IoU'])
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
        plt.savefig(os.path.join(save_path,"tick"+(str(lr))+'_train_loss'),format='png')
        plt.close()
        plt.plot(range(N_epochs),validation_loss_values, '-o')
        plt.title('Validation Loss')
        plt.xlabel('N_epochs')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(save_path,"tick"+(str(lr))+'_val_loss'),format='png')
        plt.close()




def grad_check(model,model_layers='Classifier'):
    if model_layers=='Classifier':
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(requires_grad=True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(requires_grad=True)


if __name__ == "__main__":

    #training(['model_pre_full'],path2 = r'C:\Users\Mads-_uop20qq\Documents\5. Semester\BachelorProj\Bachelorprojekt\Bachelor-Criterion-AI\semantic_segmentation\DeepLabV3\outfile.jpg')
    pass
