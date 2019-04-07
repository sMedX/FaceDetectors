## Description
This work is used for reproduce MTCNN,a Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks.

## Prerequisites
1. You need CUDA-compatible GPUs to train the model.
2. You should first download [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) and [LFW(CNN_FacePoint)](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm). **WIDER Face** for face detection and **LFW(CNN_FacePoint)** for landmark detection.

## Dependencies
* Tensorflow 1.2.1
* TF-Slim
* Python 2.7
* Ubuntu 16.04
* Cuda 8.0

## Prepare For Training Data
1. Download Wider Face Training part only from Official Website, unzip, and put the train part to the default directory `~/datasets/wider`.
2. Download landmark training data from [here]((http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)), unzip, and put the train part `(lfw_5590`, `net_7876`, `trainImageList.txt)` to the default directory `~/datasets/lfwmtcnn`.
3. Run `train_mtcnn.sh` bash file to generate training datasets and then train MTCNN.

## Some Details

## License
MIT LICENSE
