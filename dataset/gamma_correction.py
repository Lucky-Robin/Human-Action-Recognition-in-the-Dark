import cv2
import numpy as np
import os


def adjust_gamma_table(gamma):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return table


def gamma_intensity_correction(cur_path, classes, gamma):
    LU_table = adjust_gamma_table(gamma)
    # print(cur_path)     # sampled/train/Drink/
    for image_name in os.listdir(cur_path):
        image_path = os.path.join(cur_path, image_name)     # sampled/train/Drink/Drink_3_1_0.jpg
        convert_image = cv2.imread(image_path)
        image_gamma_correct = cv2.LUT(convert_image, LU_table)
        outfile = 'preprocessing/train/' + classes + '/' + image_name
        cv2.imwrite(outfile, image_gamma_correct)

    print(classes, 'Gamma Correction Finished')


# Directly and independently run gamma_correction.py to do gamma correction for training dataset
if __name__ == '__main__':
    sampled_path = 'sampled/train/'
    for sourceFileName in os.listdir(sampled_path):  # ['Drink', 'Jump', 'Pick'......
        class_path = os.path.join(sampled_path, sourceFileName + '/')   # sampled/train/Drink/
        gamma_intensity_correction(cur_path=class_path, classes=sourceFileName, gamma=5)
