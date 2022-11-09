import cv2
import os


def video_sample(sourceFileName):
    freq_step = 5  # sample every 5 frames

    class_path = os.path.join('raw/train/', sourceFileName + '/')  # raw/train/Drink/
    outdir = 'sampled/train/' + sourceFileName + '/'  # sampled/train/Drink/
    for video_file in os.listdir(class_path):  # ['Drink_12_1.mp4', 'Drink_12_2.mp4'......
        video_path = os.path.join(class_path, video_file)  # raw/train/Drink/Drink_12_1.mp4
        frame = cv2.VideoCapture(video_path)
        time = 0    # Number of frames read
        c = 0   # name of image
        if frame.isOpened():
            success, image = frame.read()
            while success:
                if time % freq_step == 0:
                    outfile = outdir + video_file[:-4] + '_' + str(c) + '.jpg'  # sampled/train/Drink/Drink_12_1_'c'.jpg
                    cv2.imwrite(outfile, image)
                    c += 1
                time += 1
                cv2.waitKey(1)
                success, image = frame.read()

    frame.release()
    print(sourceFileName, 'Sampling Finished')


# Directly and independently run video_sampler.py to sample training dataset
if __name__ == '__main__':
    train_file = 'raw/train'
    for sourceFileName in os.listdir(train_file):  # ['Drink', 'Jump', 'Pick'......
        video_sample(sourceFileName)
