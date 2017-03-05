import argparse
import json
import cPickle
import cv2
from sklearn_theano.feature_extraction import OverfeatTransformer

#parse a single command-line argument to take a json configuration file
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file...")
ap.add_argument("-i", "--image", help="path to test image...")
args = vars(ap.parse_args())

class Classifier(object):
    def __init__(self, configPath):
        self.config = json.load(open(configPath))
        self.model = cPickle.load(open(self.config["classifier_path"]))
        self.le = cPickle.load(open(self.config["label_encoder_path"]))
        self.overfeat = OverfeatTransformer(output_layers=self.config["layer_num"])

    def _preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(self.config["image_size"]))
        return image

    def _extract_overfeat_features(self, image):
        image = self._preprocess(image)
        features = self.overfeat.transform([image])
        return features

    def predict(self, imagePath):
        image = cv2.imread(imagePath)
        features = self._extract_overfeat_features(image)[0]
        prediction = self.model.predict([features])[0]
        return self.le.inverse_transform(prediction)

if __name__=="__main__":
    classifier = Classifier(args["conf"])
    prediction = classifier.predict(args["image"])
    image = cv2.imread(args["image"])
    cv2.putText(image, prediction, (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
    cv2.imshow("Image", image)
    cv2.waitKey(0)





