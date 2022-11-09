import cv2
import os


def video_sample(validation_file_path): # raw/validate
    outdir = 'sampled/validate'
    sample_num = 7  # each video only sample 7 frames (odd frames for better judge accuracy)
    for sourceFileName in os.listdir(validation_file_path):  # '0.mp4', '1.mp4',
        video_path = os.path.join(validation_file_path, sourceFileName)
        frame = cv2.VideoCapture(video_path)
        frame_num = frame.get(7)    # total frame number in this video file
        # print(sourceFileName, frame_num)
        sample_frame_num = int((frame_num - 1) / (sample_num - 1))   # sample in every sample_frame_num frame
        time = 0  # Number of frames read
        c = 0  # name of image
        if frame.isOpened():
            success, image = frame.read()
            while success:
                if time % sample_frame_num == 0:
                    outfile = outdir + '/' + sourceFileName[:-4] + '_sampled_' + str(c) + '.jpg'  # sampled/validate/0_sampled_'c'.jpg
                    cv2.imwrite(outfile, image)
                    c += 1
                time += 1
                cv2.waitKey(1)
                success, image = frame.read()
        if c != 7:  # 33.mp4, 276.mp4 has additional frames, manually delete
            print(sourceFileName, c)
        # print(sourceFileName, 'Sampling Finished')
    frame.release()
    print('Sampling Finished')


# Directly and independently run validation_video_sampler.py to sample validation dataset
if __name__ == '__main__':
    validation_file_path = 'raw/validate'
    video_sample(validation_file_path)
