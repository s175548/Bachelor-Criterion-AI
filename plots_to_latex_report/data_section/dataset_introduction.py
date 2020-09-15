from data_import.data_loader import DataLoader
from data_import.masks_to_bounding_box import get_background_mask,combine_seg_and_back_mask

import cv2,numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    ### PLOT 1 #### - The 3 defects, Piega, Verruca, Area Punture Insetti
    # dataloader = DataLoader()
    # img_piega, mask_piega = dataloader.get_image_and_labels([23, 24, 63, 64, 113, 114, 115])
    # img_verruca, mask_verruca = dataloader.get_image_and_labels(list(range(44, 59)) + [100, 101])
    # img_area_punture_insetti, mask_area_punture_insetti = dataloader.get_image_and_labels([488, 490, 518])
    #
    # dataloader.plot_function(img_piega, mask_piega)
    # dataloader.plot_function(img_verruca, mask_verruca)
    # dataloader.plot_function(img_area_punture_insetti, mask_area_punture_insetti)

    ### PLOT 2 #### Mask plot - With original image, original mask, background mask and combined mask as an overlay on top of the original image.

    # data_loader = DataLoader()
    # img,mask = data_loader.get_image_and_labels(1)
    # back_mask = get_background_mask(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # combined_mask = combine_seg_and_back_mask([1],data_loader)
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].imshow(img)
    # axarr[0, 0].set_title('Original image')
    # axarr[0, 1].imshow(mask,cmap="Greys")
    # axarr[0, 1].set_title('Original mask')
    # axarr[1, 0].imshow(back_mask,cmap="Blues")
    # axarr[1, 0].set_title('Background mask')
    # axarr[1, 1].imshow(img)
    # axarr[1, 1].imshow(mask,cmap="Greys",alpha = 0.3)
    # axarr[1, 1].imshow(back_mask, cmap="Blues",alpha=0.3)
    # axarr[1,1].set_title('Original image with both masks')
    # [axi.set_axis_off() for axi in axarr.ravel()]
    # plt.show()

### PLOT 3 ###
    data_loader = DataLoader()
    img,mask = data_loader.get_image_and_labels(1)
    back_mask,mask1_nofilter = get_background_mask(img) #Return mask1 from get_background_mask() to get the correct output
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(mask1_nofilter[:256,256:],cmap="Blues")
    axarr[0].set_title('Original image')
    axarr[1].imshow(back_mask[:256,256:],cmap="Blues")
    axarr[1].set_title('Original mask')
    [axi.set_axis_off() for axi in axarr.ravel()]
    plt.show()
