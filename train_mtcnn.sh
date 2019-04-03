#!/bin/bash 
echo "train mtcnn"

# input datasets directory
ds=~
echo "datasets directory" ${ds}

# output logs and models directories
md=~
echo "output directory" ${md}

python3 tfmtcnn/apps/train_mtcnn.py \
	--wider ${ds}/datasets/wider \
	--lfw   ${ds}/datasets/lfw \
	--mtcnn ${md}/mtcnn

