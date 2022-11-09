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


def gamma_intensity_correction(cur_path, gamma):
    LU_table = adjust_gamma_table(gamma)
    for image_name in os.listdir(cur_path):
        image_path = os.path.join(cur_path, image_name)     # sampled/validate/0_sampled_0.jpg
        convert_image = cv2.imread(image_path)
        image_gamma_correct = cv2.LUT(convert_image, LU_table)
        outfile = 'preprocessing/validate/' + image_name    # preprocessing/validate/0_sampled_0.jpg
        cv2.imwrite(outfile, image_gamma_correct)
        print(image_name, 'Gamma Correction Finished')


# Directly and independently run validation_gamma_correction.py to do gamma correction for validation dataset
if __name__ == '__main__':
    sampled_path = 'sampled/validate/'
    gamma_intensity_correction(cur_path=sampled_path, gamma=5)
