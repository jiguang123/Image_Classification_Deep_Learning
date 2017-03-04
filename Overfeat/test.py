from __future__ import print_function
import json
import numpy as np
import argparse
import cPickle
import h5py
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file...")
args = vars(ap.parse_args())

print("[INFO] loading model...")
config = json.load(open(args["conf"]))
le = cPickle.load(open(config["label_encoder_path"]))
model = cPickle.load(open(config["classifier_path"]))

print("[INFO] gathering test data...")
db = h5py.File(config["features_path"])
split = int(db["image_ids"].shape[0] * config["training_size"])
(testData, testLabels) = (db["features"][split:],db["image_ids"][split:])

for i in np.random.choice(np.arange(0, len(testData)),size=(10,)):
    (trueLabel, filename) = testLabels[i].split(":")
    vector = testData[i]

    prediction = model.predict(np.atleast_2d(vector))[0]
    prediction = le.inverse_transform(prediction)
    print("[INFO] predicted: {}, actual: {}".format(prediction, trueLabel))

    path = "{}/{}/{}".format(config["dataset"],trueLabel,filename)
    image = cv2.imread(path)
    cv2.putText(image, prediction, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.imshow("Image {}".format(i), image)
cv2.waitKey(0)
db.close()
