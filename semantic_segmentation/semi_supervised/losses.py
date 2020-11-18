'''
  the author is leilei;
  Loss functions are in here.
  分别计算 有标签真实数据损失函数、生成数据损失函数、无标签真实数据损失函数。
'''
import numpy as np,torch
import torch.nn.functional as F
def log_sum_exp(x,axis=1):
    '''
    Args:
        x : [n*h*w,c],semantic segmentation‘s output’s shape is [n,c,h,w]，before input need to reshape [n*h*w,c]
    '''
    m = torch.max(x,dim=axis)[0]
    return m+torch.log(torch.sum(torch.exp(x-torch.unsqueeze(m,dim=axis)),dim=axis))

def Loss_label(pred,label):
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    label: [n,h,w] ,tensor need to numpy ,then need to reshape [n*h*w,1]
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_lab = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]])# [n*h*w, c]
    
    label_ = label.data.cpu().numpy().reshape([-1,]) #MADS _ giver forkert dim.
    label_ = np.expand_dims(label.data.cpu().numpy().reshape([-1,]), axis=1) #Vi udvider:
    # l_lab before softmax
    l_lab = output_before_softmax_lab[np.arange(label_.shape[0]),label_]
    # compute two value
    loss_lab = -torch.mean(l_lab) + torch.mean(log_sum_exp(output_before_softmax_lab))
    
    return loss_lab

def Loss_fake(pred):
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_gen = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]])# [n*h*w, c]
    l_gen = log_sum_exp(output_before_softmax_gen)
    loss_gen = torch.mean(F.softplus(l_gen))
    
    return loss_gen

def Loss_unlabel(pred):
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c] 
    '''
    shape = pred.shape# n c h w
    # predict before softmax
    output_before_softmax_unl = pred.transpose(1,2).transpose(2,3).reshape([-1,shape[1]])# [n*h*w, c]
    
    l_unl = log_sum_exp(output_before_softmax_unl)
    loss_unl = -torch.mean(l_unl) + torch.mean(F.softplus(l_unl))
    
    return loss_unl



############# Own implementations:
def Loss_unlabel_remade(pred):
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c]
    '''
    only_one_conf_layer = True
    criterion_ul = torch.nn.BCELoss()
    real_label = 0.
    shape = pred.detach().cpu().numpy().shape
    #shape = pred.shape  # n c h w  [2, 4 , 256 , 256]
    # # predict before softmax
    output_before_softmax_unl = pred.transpose(1, 2).transpose(2, 3).reshape([-1, shape[1]])  # [n*h*w, c] [131072,4]
    soft = torch.nn.Softmax(dim=1)
    confidence_map_prob = soft(output_before_softmax_unl)
    #
    fake_confidence_map = confidence_map_prob[:,3] #[131072] pixels from both images in batch
    #
    # #Convert to probabilities :
    # loss_unl = torch.sum(torch.mean(torch.log(1-fake_confidence_map)))

    label = torch.full((shape[0],shape[2],shape[3]), real_label, dtype=torch.float).to('cuda')
    label = label.reshape([-1])
    label.fill_(real_label)
    #pred_fake_confidence_map = (pred[:,shape[1]-1,:,:] - torch.min(pred[:,shape[1]-1,:,:]) ) / (torch.max(pred[:,shape[1]-1,:,:]) -torch.min(pred[:,shape[1]-1,:,:]) )
    loss_unl = criterion_ul(fake_confidence_map, label)
    if only_one_conf_layer:
        fake_layer = output_before_softmax_unl[:,3]
        std_fake_layer = (fake_layer-torch.min(fake_layer)) / (torch.max(fake_layer)-torch.min(fake_layer))
        loss_gen = criterion_ul(std_fake_layer, label)
    return loss_unl


def Loss_fake_remade(pred):
    only_one_conf_layer = True
    fake_label = 1.
    '''
    pred: [n,c,h,w],need to transpose [n,h,w,c],then reshape [n*h*w,c]
    '''
    criterion_fake = torch.nn.BCELoss()
    shape = pred.detach().cpu().numpy().shape
    #shape = pred.shape  # n c h w
    # predict before softmax
    output_before_softmax_gen = pred.transpose(1, 2).transpose(2, 3).reshape([-1, shape[1]])  # [n*h*w, c]
    soft = torch.nn.Softmax(dim=1)
    confidence_map_prob = soft(output_before_softmax_gen)
    fake_confidence_map = confidence_map_prob[:, 3]  # [131072] pixels from both images in batch
    #
    #
    # loss_gen = torch.sum( torch.mean(torch.log(1-(1-fake_confidence_map))) )
    label = torch.full((shape[0],shape[2],shape[3]), fake_label, dtype=torch.float).to('cuda')
    label.fill_(fake_label)
    #pred_fake_confidence_map = (pred[:,shape[1]-1,:,:] - torch.min(pred[:,shape[1]-1,:,:]) ) / (torch.max(pred[:,shape[1]-1,:,:]) -torch.min(pred[:,shape[1]-1,:,:]) )
    label = label.reshape([-1])
    loss_gen = criterion_fake(fake_confidence_map, label)
    if only_one_conf_layer:
        print("ONLY SOFTMAX ON")
        fake_layer = output_before_softmax_gen[:,3]
        std_fake_layer = (fake_layer-torch.min(fake_layer)) / (torch.max(fake_layer)-torch.min(fake_layer))
        loss_gen = criterion_fake(std_fake_layer, label)

    return loss_gen