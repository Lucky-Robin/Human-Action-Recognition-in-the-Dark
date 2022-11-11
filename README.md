# Human Action Recognition in the Dark
This project is an Assignment for the EE6222 Machine Vision course at Nanyang Technological University. 
The project requires to propose a method to perform Human Action Recognition (HAR) in videos shot in the dark. 
In this continuous assessment, the task is restricted to HAR in a trimmed video dataset (ARID Dataset) with one action 
per video given RGB frames.


## Install
The following version of the package is used for this project and has also been placed in requirements.txt.
```
numpy==1.23.4
opencv_python==4.6.0.66
Pillow==9.3.0
torch==1.13.0
torchvision==0.14.0
```

## Usage
Sampling:  
* For training dataset, change path and run \dataset\video_sampler.py.  
* For validation dataset, change path and run \dataset\validation_video_sampler.py.  
* For test dataset, change path and un \dataset\test_video_sampler.py.  

Gamma Correction:  
* For training dataset, change path and run \dataset\gamma_correction.py.  
* For validation dataset, change path and run \dataset\validation_gamma_correction.py.  
* For test dataset, change path and run \dataset/test_gamma_correction.py.  

Training and Testing:
* For model training and validation purpose, change path and run \train.py
* For generating testing results purpose, change path and run \test.py. It will generate an /Test_output.txt file, 
containing testing video number and their classifications.  

