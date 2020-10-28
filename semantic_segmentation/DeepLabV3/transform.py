transform_function = et.ExtCompose([et.ExtRandomCrop(size=2048),
                                    et.ExtResize(scale=0.25)
                                    et.ExtRandomCrop(size=0.7)
                                    et.ExtRandomHorizontalFlip(p=0.5),
                                    et.ExtRandomVerticalFlip(p=0.5),
                                    et.ExtEnhanceContrast(),
                                    et.ExtToTensor(),
                                    et.ExtNormalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])