import cv2 as cv
import PIL
import numpy as np

transform_function = et.ExtCompose([et.ExtTransformLabel(),et.ExtCenterCrop(512),et.ExtScale(200),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),])

pil_img=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/img/100.jpg')
pil_mask=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/mask/100_mask.png')
img,_ = transform_function(pil_img,pil_mask)


im = cv.imread('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/mask/10_mask.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
imgray=255-imgray
ret, thresh = cv.threshold(imgray, 127, 255, 0)
img, contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
mask=cv.drawContours(im,contours,1,color=(0,255,0),thickness=cv.FILLED)
mask[:,:,2]==255
img=PIL.Image.fromarray(mask)
img.show()

model=deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
model.classifier[-1]=torch.nn.Conv2d(256,2,kernel_size=(1,1),stride=(1,1)).requires_grad_()
model.aux_classifier[-1]=torch.nn.Conv2d(256,2,kernel_size=(1,1),stride=(1,1)).requires_grad_()
checkpoint = torch.load('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /model_pre_class.pt', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["model_state"])

path_img='/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/img/'
file_names = np.array([img[:-4] for img in os.listdir(path_img)])
N_files = len(file_names)
shuffled_index = np.random.permutation(len(file_names))
file_names_img = file_names[shuffled_index]
file_names = file_names[file_names != ".DS_S"]
file_names=file_names[:30]
model.eval()

for file in file_names:
    pil_img=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/img/'+file+'.jpg')
    pil_mask=PIL.Image.open('/Users/villadsstokbro/Dokumenter/DTU/KID/5. Semester/Bachelor /data_folder/mask/100_mask.png')
    img,_ = transform_function(pil_img,pil_mask)
    output=model(img.unsqueeze(0))
    output_new=output['out']
    preds = output_new.detach().max(dim=1)[1].cpu().numpy()
    PIL.Image.fromarray(preds.resize(200,200)+1)
