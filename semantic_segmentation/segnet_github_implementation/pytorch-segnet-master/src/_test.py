"""Test for SegNet"""

from __future__ import print_function
from model import SegNet
from dataset import NUM_CLASSES
import matplotlib.pyplot as plt
import numpy as np
import torch

import cv2


def load_image(self, path=None):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,224), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('image', raw_image)
    cv2.waitKey(0)
    raw_image = np.expand_dims(raw_image.transpose(2, 0,1), axis=0)

    imx_t = np.array(raw_image, dtype=np.float32) / 255.0
    return imx_t
if __name__ == "__main__":
    # RGB input
    input_channels = 3
    # RGB output
    output_channels = NUM_CLASSES

    # Model
    model = SegNet(input_channels=input_channels, output_channels=output_channels)

    #print(model)

    img = torch.randn([4, 3, 224, 224])

    # plt.imshow(np.transpose(img.numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()
    #image_path = r'C:\Users\Mads-\Desktop\leather_patches\RED_HALF02\Grain\RED_HALF02_grain_01_v.tif\Abassamento\4532x1152.png'
    image_path = r'C:\Users\Mads-\Desktop\Left_side_of_Flying_Pigeon.jpg'
    image_test = load_image(image_path)


    pass

    #cv2.imshow('image', image_test1)
    #cv2.waitKey(0)
    output, softmaxed_output = model(torch.from_numpy(image_test))

    #output, softmaxed_output = model(img)


    # plt.imshow(np.transpose(output.detach().numpy()[0,:,:,:],
    #                         (1, 2, 0)))
    # plt.show()


    print(output.size())
    print(softmaxed_output.size())

    print(output[0,:,0,0])
    print(softmaxed_output[0,:,0,0].sum())
    print(np.where(max(softmaxed_output[0,:,0,0].sum())==softmaxed_output[0,:,0,0].sum()))

