# Image Classification with Deep Learning
This script is created to support the video session at 

### Links
MNIST datasets: https://pjreddie.com/projects/mnist-in-csv/

CALTECH-101: http://www.vision.caltech.edu/Image_Datasets/Caltech101/

### Usage:
Extracting features:
`python extract_features.py --conf <path_to_configuration_json_file>`

Training model:
`python train.py --conf <path_to_configuration_json_file>`

Test model:
`python test.py --conf <path_to_configuration_json_file>`

Classify new images
`python classify.py --conf <path_to_configuration_json_file> --image <path_to_image>`


