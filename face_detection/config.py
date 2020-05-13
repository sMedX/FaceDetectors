from pathlib import Path

root_dir = Path(__file__).parents[1]

data_dir = root_dir.joinpath('data')
dir_images = data_dir.joinpath('images')
dir_output = data_dir.joinpath('output')

# detectors
default_detector = 'ssdlite_mobilenet_v2_coco_2018_05_09'

