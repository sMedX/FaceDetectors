#export CUDA_VISIBLE_DEVICES=0
#export PYTHONPATH=$PYTHONPATH:/media/b/repo/tf_models/research:/media/b/repo/tf_models/research/slim

md=/home/korus/workspace/faces/venv-objdet/lib/python3.6/site-packages/tensorflow/models/research

python3 $md/object_detection/legacy/train.py \
 --logtostderr \
 --train_dir=output \
 --pipeline_config_path=configs/ssd_inception_v2_coco/ssd_inception_v2_coco.config
