from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
from threading import Thread;
import sys
import time
import datetime
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2
import imutils
from object_detection import ObjectDetection


# resources: https://www.youtube.com/watch?v=TX4MrMNYkXs&ab_channel=JonWood
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
# https://docs.microsoft.com/en-us/azure/cognitive-services/custom-vision-service/quickstarts/object-detection?tabs=visual-studio&pivots=programming-language-python
#https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()

        with self.graph.as_default():
            tf.compat.v1.disable_eager_execution()
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]

class FileVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped =False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while True:
            if self.stopped:
                return
            
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        
graph = tf.compat.v1.GraphDef()
labels = []


filename = "model.pb"
labels_file = "labels.txt"
output_layer = 'loss:0'
input_node = 'Placeholder:0'

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph.ParseFromString(f.read())
    tf.import_graph_def(graph, name='')

with open(labels_file, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())


#cam = cv2.VideoCapture(0)
od_model = TFObjectDetection(graph, labels)
videoStream = FileVideoStream(src=0).start()

while True:
    img = videoStream.read()
    img = cv2.resize(img, (1920,1080), interpolation = cv2.INTER_AREA)
    
    predictions = od_model.predict_image(img)
    for p  in predictions:
        if p["tagName"] == "not a subscriber":
            color = (0,0,255)
        else:
            color = (0,255,0)

        x = int(p["boundingBox"]["left"]*img.shape[1])
        y= int(p["boundingBox"]["top"]*img.shape[0])
        w = int(p["boundingBox"]["width"]*img.shape[1])
        h = int(p["boundingBox"]["height"]*img.shape[0])
        img = cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
        img = cv2.putText(img, p["tagName"], (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 1, cv2.LINE_8, False)

    cv2.imshow('img', img)
    key = cv2.waitKey(1) & 0xff
    # if key ==27:
    #     break
    
videoStream.stop()
cv2.destroyAllWindows()
#cam.release()