"""
Made with heavy inspiration from
https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/af50e37932732a2c06e331c54cc8c64820c307f4/main.py
"""

from tqdm import tqdm
import random
import numpy as np
from semantic_segmentation.DeepLabV3.dataset_class import LeatherData

from torch.utils import data
from semantic_segmentation.DeepLabV3.metrics import StreamSegMetrics
from semantic_segmentation.DeepLabV3.utils import ext_transforms as et
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import os



model=deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
for param in model.parameters():
    param.requires_grad = False

model.classifier[-1]=torch.nn.Conv2d(256,2,kernel_size=(1,1),stride=(1,1)).requires_grad_()
model.aux_classifier[-1]=torch.nn.Conv2d(256,2,kernel_size=(1,1),stride=(1,1)).requires_grad_()

transform_function = et.ExtCompose([et.ExtTransformLabel(),et.ExtCenterCrop(512),et.ExtScale(100),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),])

num_classes=2
output_stride=16
save_val_results=False
total_itrs=100
lr=0.01
lr_policy='step'
step_size=10000
batch_size=16
val_batch_size=4
loss_type="cross_entropy"
weight_decay=1e-4
random_seed=1
print_interval=10
val_interval=20
vis_num_samples=2
enable_vis=True


path_mask = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/mask'
path_img = r'/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/img'



def save_ckpt(path='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /DeepLabV3_finetuned.pt',cur_itrs=None, optimizer=None,scheduler=None,best_score=None):
    """ save current model
    """
    torch.save({
        "cur_itrs": cur_itrs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_score": best_score,
    }, path)
    print("Model saved as %s" % path)

def validate(model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)['out']
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

        score = metrics.get_results()
        print(score)
    return score, ret_samples



def main(model=None):

    # Setup visualization
    best_score = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    # Setup dataloader
    file_names = np.array([img[:-4] for img in os.listdir(path_img)])
    N_files=len(file_names)
    shuffled_index=np.random.permutation(len(file_names))
    file_names_img=file_names[shuffled_index]
    file_names=file_names[file_names != ".DS_S"]

    train_dst = LeatherData(path_mask=path_mask,path_img=path_img,list_of_filenames=file_names_img[:round(N_files*0.80)], transform=transform_function)
    val_dst = LeatherData(path_mask=path_mask, path_img=path_img,list_of_filenames=file_names_img[round(N_files*0.80):], transform=transform_function)
    train_loader = data.DataLoader(
        train_dst, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=val_batch_size, shuffle=True, num_workers=2)

    print("Train set: %d, Val set: %d" %
          (len(train_dst), len(val_dst)))

    # Set up model
    model = model

    # Set up metrics
    metrics = StreamSegMetrics(num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.classifier.parameters(), 'lr': lr},
    ], lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')



    # ==========   Train Loop   ==========#

    interval_loss = 0
    cur_epochs=0
    while cur_epochs<4 :  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_itrs=0
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)['out']

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            print('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 1 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % val_interval == 0:
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    model=model, loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(cur_itrs=cur_itrs, optimizer=optimizer, scheduler=scheduler, best_score=best_score)
                    print("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    print("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    print("[Val] Class IoU", val_score['Class IoU'])
                model.train()
            scheduler.step()

            if cur_itrs >= total_itrs:
                return
main(model)