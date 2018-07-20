import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from grabscreen import grab_screen
import cv2

#This is needed since the Jupyter notebook is stored in the object_detection folder
sys.path.append("..")

#Imports from the object_detection folder under models
from utils import label_map_util
from utils import visualization_utils as vis_util

#MODEL PREPERATION 
#Change here what model needs to be downloaded
MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

#Path to frozen detection graph.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

#Lists all strings used to add labels for each object detection boxes
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

#Downloading the Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

#Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#Loading label map
#Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

#MAIN SECTION OF THE PROGRAM
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #800x640 -> This is the resolution I am currently running the game on
      #screen = cv2.resize(grab_screen(region=(0,40,800,640)), (WIDTH,HEIGHT))
      screen = cv2.resize(grab_screen(region=(0,70,800,640)), (800,450))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      #Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      #Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      #Each score represent how level of confidence for each of the objects.
      #Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      #Actual Object_Detection
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #Visualization of the results of a detection (Draws the boxes)
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      #Calculating proximity based on the distance between boxes
      for i, b in enumerate(boxes[0]):
        #This initializes the classes with the size of car, truck and bus respectively
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][2]:
          #The score tells us what percentage possibility that a crash is going to occur
          if scores[0][i] > 0.5:
            #mid_x, mid_y and apx_distance are all percentages NOT pixels
            mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
            mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
            apx_distance = round( (1 -(boxes[0][i][3] - boxes[0][i][1]))**3,1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800), int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            #This executes only if the car is really close. 0.8 can be changed based on future observations of performance in game
            if apx_distance <= 0.8:
              if mid_x > 0.3 and mid_x <= 0.7:
                cv2.putText(image_np, 'WATCH OUT!', (int(mid_x*500)-50, int(mid_y*700)), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)
            if apx_distance <= 0.4:
             if mid_x > 0.3 and mid_x <= 0.7:
               cv2.putText(image_np, 'CRASH ALERT!', (int(mid_x*800)-50, int(mid_y*450)), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0,0,255), 3)
        

      cv2.imshow('window',image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
