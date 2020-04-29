from pathlib import Path

root_dir = Path(__file__).parents[1]

data_dir = root_dir.joinpath('data')
dir_images = data_dir.joinpath('images')
dir_output = data_dir.joinpath('output')

# detectors
ssdlite_mobilenet = 'ssdlite_mobilenet'
ssd_inception_resnet = 'ssd_inception_resnet'
ssd_inception_v2_coco = 'ssd_inception_v2_coco'

