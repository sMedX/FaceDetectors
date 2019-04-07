#!/bin/bash 
echo "train mtcnn"

# input datasets directory
ds=~
echo "datasets directory" "${ds}"

# output logs and models directories
md=~
echo "output directory" "${md}"

mtcnn_train \
    --wider "${ds}/datasets/wider" \
    --lfw   "${ds}/datasets/lfwmtcnn" \
    --mtcnn "${md}/mtcnn"
