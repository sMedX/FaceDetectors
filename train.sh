# export CUDA_VISIBLE_DEVICES=0

#export PYTHONPATH=$PYTHONPATH:/home/korus/workspace/tf_models/research
#export PYTHONPATH=$PYTHONPATH:/home/korus/workspace/tf_models/research/slim
python3 -m object_detection.legacy.train --logtostderr --train_dir=training/ --pipeline_config_path=configs/ssd_inception_v2_coco.config
# python3 -m home.korus.workspace.tf_models.research.object_detection.legacy.train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_inception_v2_coco.config