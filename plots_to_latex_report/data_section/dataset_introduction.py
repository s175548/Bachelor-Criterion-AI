from data_import.data_loader import DataLoader
import cv2,numpy as np


if __name__ == "__main__":
    dataloader = DataLoader()
    img_piega, mask_piega = dataloader.get_image_and_labels([23, 24, 63, 64, 113, 114, 115])
    img_verruca, mask_verruca = dataloader.get_image_and_labels(list(range(44, 59)) + [100, 101])
    img_area_punture_insetti, mask_area_punture_insetti = dataloader.get_image_and_labels([488, 490, 518])

    dataloader.plot_function(img_piega, mask_piega)
    dataloader.plot_function(img_verruca, mask_verruca)
    dataloader.plot_function(img_area_punture_insetti, mask_area_punture_insetti)