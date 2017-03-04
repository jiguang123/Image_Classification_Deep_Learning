from __future__ import print_function
from imutils.paths import list_images
from sklearn_theano.feature_extraction import OverfeatTransformer
from sklearn_theano.feature_extraction.overfeat import SMALL_NETWORK_FILTER_SHAPES
import h5py
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cPickle
import argparse
import random
import json
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file...")
args = vars(ap.parse_args())

print("[INFO] loading configurations...")
config = json.loads(open(args["conf"]).read())
imagePaths = list(list_images(config["dataset"]))

output_size = SMALL_NETWORK_FILTER_SHAPES[config["layer_num"]][0]
batch_size = config["batch_size"]

print("[INFO] creating datasets...")
db = h5py.File(config["features_path"], mode="w")
imageIDDB = db.create_dataset("image_ids",shape=(len(imagePaths),), dtype=h5py.special_dtype(vlen=unicode))
featuresDB = db.create_dataset("features", shape=(len(imagePaths), output_size), dtype="float")

random.seed(42)
random.shuffle(imagePaths)

print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([p.split("/")[-2] for p in imagePaths])

print("[INFO] initializing network...")
overfeat = OverfeatTransformer(output_layers=config["layer_num"])

print("[INFO] extracting features...")

for start in xrange(0,len(imagePaths), batch_size):
    end = start + batch_size

    #read and resize the images
    images = [cv2.imread(impath) for impath in imagePaths[start:end]]
    images = np.array([cv2.resize(image, tuple(config["image_size"])) for image in images], dtype="float")

    #dump the image ID and features to the hdf5 database
    imageIDDB[start:end] = [":".join(impath.split("/")[-2:]) for impath in imagePaths[start:end]]
    features = overfeat.transform(np.array(images))
    featuresDB[start:end] = overfeat.transform(np.array(images))
    print("[INFO] processed {} images".format(end))

db.close()
f = open(config["label_encoder_path"], "w")
cPickle.dump(le, f)
f.close()
