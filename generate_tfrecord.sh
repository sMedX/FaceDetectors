
python3 -m face_detection.apps.generate_tfrecord \
  --remove_invalid True \
  --minsize 12 \
  --txt_input   /home/korus/datasets/wider/wider_face_split/wider_face_train_bbx_gt.txt \
  --img_path    /home/korus/datasets/wider/WIDER_train/images \
  --output_path /home/korus/datasets/wider/train.tfrecord

python3 -m face_detection.apps.generate_tfrecord \
  --remove_invalid True \
  --minsize 12 \
  --txt_input   /home/korus/datasets/wider/wider_face_split/wider_face_val_bbx_gt.txt \
  --img_path    /home/korus/datasets/wider/WIDER_val/images \
  --output_path /home/korus/datasets/wider/test.tfrecord