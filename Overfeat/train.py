from __future__ import print_function
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import h5py
import argparse
import json
import cPickle

#parse a single command-line argument to take a json configuration file
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file...")
args = vars(ap.parse_args())

print("[INFO] loading configurations, data, encoder ...")
config = json.load(open(args["conf"]))
db = h5py.File(config["features_path"], mode="r")
le = cPickle.load(open(config["label_encoder_path"]))

#split the training and testing datasets
split = int(db["image_ids"].shape[0]*config["training_size"])
(trainData, trainLabels) = (db["features"][:split], db["image_ids"][:split])
(testData, testLabels) = (db["features"][split:], db["image_ids"][split:])

#encode the training and testing labels
trainLabels = np.array([le.transform([l.split(":")[0]]) for l in trainLabels])
testLabels = np.array([le.transform([l.split(":")[0]]) for l in testLabels])

print("[INFO] training model...")
model = LogisticRegression(C=0.1)
model.fit(trainData, trainLabels.flatten())

print("[INFO] evaluating model...")

# Rank 1: If the actual target is equal to the top predicted class
# Rank 5: If the actual target is in top five predicted classes

rank1, rank5 = 0, 0
for (label, features) in zip(testLabels, testData):
    preds = model.predict_proba(np.atleast_2d(features))[0]
    preds = np.argsort(preds)[::-1][:5]

    if label == preds[0]:
        rank1 += 1

    if label in preds:
        rank5 += 1

rank1 = (rank1/float(len(testLabels)))*100
rank5 = (rank5/float(len(testLabels)))*100

#predict the testing data
predictions = model.predict(testData)

print("Rank 1:", rank1)
print("Rank 5: ", rank5)
print(classification_report(testLabels, predictions, target_names=le.classes_))

f = open(config["classifier_path"], "w")
f.write(cPickle.dumps(model))
f.close()

db.close()
