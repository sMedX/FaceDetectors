export CUDA_VISIBLE_DEVICES=1

md=/home/ruslan/Faces/venv-faces/lib/python3.6/site-packages/tensorflow/models/research

python3 $md/object_detection/legacy/train.py \
 --logtostderr \
 --train_dir=output \
 --pipeline_config_path=configs/ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03/pipeline.config
