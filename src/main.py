
from utils.image_classifier import ImageClassifier, NO_FACE_LABEL

import cv2.cv2 as cv2
import pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3
from keras.models import load_model
from clahe import clahe

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
prediction = None



#trained model for sign-language detection

model = load_model('cnn_model_keras2.h5')

# Color RGB Codes & Font

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 255, 104)
FONT = cv2.QT_FONT_NORMAL

# Frame Width & Height

FRAME_WIDTH = 640
FRAME_HEIGHT = 490

#link to camera feed (set to uri = '0' for webcam)
uri = '0'

def emote(label):
    if label == "no face":
        return (255,255,255), "has no face"
    elif label == "neutral":
        return (255,255,255), "is bored"
    elif label == "anger":
        return (0,0,255), "is angry"
    elif label == "happy":
        return (0,100,255), "is happy"
    elif label == "surprise":
        return (255,255,0), "is surprised"
    elif label == "sadness":
        return (255,0,0), "is sad"
    elif label == "disgust":
        return (0,255,0), "is disgusted"
    elif label == "fear":
        return (255,0,100), "is scared"
    return (255,255,255), "none"
    


def get_image_size():
	img = cv2.imread('gestures/1/100.jpg', 0)
	return img.shape

image_x, image_y = (50,50)

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

#sign language prediction helpers

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

#text helper functions

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]


	
def split_sentence(text, num_of_words):
	
	list_words = text.split(" ")
	length = len(list_words)
  # making patches of words of length num_of_words
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text, color):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, color)
		y += 50



class BoundingBox:
  def __init__(self, x, y, w, h):
    self.x = x
    self.y = y
    self.w = w
    self.h = h

  @property
  def origin(self) -> tuple:
    return self.x, self.y
  @property
  def top_right(self) -> int:
    return self.x + self.w
  @property
  def bottom_left(self) -> int:
    return self.y + self.h

# given an image and its bounding bow we make a rectangle that is added to the image
def draw_face_rectangle(bb: BoundingBox, img, color=BLUE_COLOR):
    cv2.rectangle(img, bb.origin, (bb.top_right, bb.bottom_left), color, 2)

# 
def draw_landmark_points(points: np.ndarray, img, color=WHITE_COLOR):
  if points is None:
    return None
  for (x, y) in points:
    cv2.circle(img, (x, y), 1, color, -1)

# what is label.upper
def write_label(x: int, y: int, label: str, img, color=BLUE_COLOR):
  if label == NO_FACE_LABEL:
    cv2.putText(img, label.upper(), (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)), FONT, 1, color, 2, cv2.LINE_AA)
  cv2.putText(img, label, (x + 10, y - 10), FONT, 1, color, 2, cv2.LINE_AA)

# Contrast Limited Adaptive Histogram Equalization
class RealTimeEmotionDetector:
  #CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  vidCapture = None
  def __init__(self, classifier_model: ImageClassifier):
    self.__init_video_capture(frame_w=FRAME_WIDTH, frame_h=FRAME_HEIGHT)
    self.classifier = classifier_model
  
  def __init_video_capture(self, frame_w: int, frame_h: int):
    self.vidCapture = cv2.VideoCapture(0)
    self.vidCapture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    self.vidCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    
  def read_frame(self) -> np.ndarray:
    rect, frame = self.vidCapture.read()
    return frame
  
  
  def transform_img(self, img: np.ndarray) -> np.ndarray:
    # load the input image, resize it, and convert it to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to gray-scale
    resized_img = clahe(gray_img,8)  # resize
    print(type(resized_img))
    return resized_img

  def recognize(self):
    global prediction
    cam = cv2.VideoCapture(0)
    if cam.read()[0] == False:
      cam = cv2.VideoCapture(0)
    hist = get_hand_hist()
    x, y, w, h = 300, 100, 300, 300
    # emotion **
    frame_cnt = 0
    predicted_labels = ['']
    old_txt = None
    rectangles = [(0, 0, 0, 0)]
    landmark_points_list = [[(0, 0)]]
    wait_key_delay=33
    quit_key='q'
    frame_period_s=0.75
    # emotion **
    text = ""
    
    #main loop
    
    while True:
      im = self.read_frame()
      im = cv2.flip(im, 1)
      frame_cnt += 1
      
      #emotion prediction every fixed number of frames    
      
      if frame_cnt % (frame_period_s * 100) == 0:
        predicted_labels = self.classifier.classify(img=self.transform_img(img=im))
        rectangles = self.classifier.extract_face_rectangle(img=im)
        landmark_points_list = self.classifier.extract_landmark_points(img=im)
        
      # classifier is a class written in another piece of code
      
      for lbl, rectangle, lm_points in zip(predicted_labels, rectangles, landmark_points_list):
        draw_face_rectangle(BoundingBox(*rectangle), im)
        draw_landmark_points(points=lm_points, img=im)
        write_label(rectangle[0], rectangle[1], label=lbl, img=im)
        if old_txt != predicted_labels:
        
          # if the emotion has got changed
          
          print('[INFO] Predicted Labels:', predicted_labels)
          old_txt = predicted_labels
          
      # sign part
      
      im = cv2.resize(im, (640, 480))
      imgCrop = im[y:y+h, x:x+w]
      imgHSV = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
      
      # Back project to get hand detection
      
      dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
      disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
      cv2.filter2D(dst,-1,disc,dst)
      blur = cv2.GaussianBlur(dst, (11,11), 0)
      blur = cv2.medianBlur(blur, 15)
      thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      thresh = cv2.merge((thresh,thresh,thresh))
      thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
      thresh = thresh[y:y+h, x:x+w]
      (openCV_ver,_,__) = cv2.__version__.split(".")
      if openCV_ver=='3':
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
      elif openCV_ver=='4':
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
      if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        #print(cv2.contourArea(contour))
        if cv2.contourArea(contour) > 10000:
          x1, y1, w1, h1 = cv2.boundingRect(contour)
          save_img = thresh[y1:y1+h1, x1:x1+w1]
          
          if w1 > h1:
            save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
          elif h1 > w1:
            save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
            
          pred_probab, pred_class = keras_predict(model, save_img)
          if pred_probab*100 > 80:
            text = get_pred_text_from_db(pred_class)
            print(text)

      col, lab = emote(predicted_labels[0])      
      blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
      splitted_text = split_sentence(text + " " + lab, 2)
      put_splitted_text_in_blackboard(blackboard, splitted_text,col)
      #cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
      cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 2)
      res = np.hstack((im, blackboard))
      cv2.imshow("Recognizing gesture", res)
      cv2.imshow("thresh", thresh)
      if cv2.waitKey(1) == ord('q'):
        break

# driver function

def run_real_time_emotion_detector(
        classifier_algorithm: str,
        predictor_path: str,
        dataset_csv: str,
        dataset_images_dir: str = None):
  from utils.data_land_marker import LandMarker
  from utils.image_classifier import ImageClassifier
  from os.path import isfile
  
  land_marker = LandMarker(landmark_predictor_path=predictor_path)

  if not isfile(dataset_csv): 
    # If data-set not built before.
    print('[INFO]', f'Dataset file: "{dataset_csv}" could not found.')
    from data_preparer import run_data_preparer
    run_data_preparer(land_marker, dataset_images_dir, dataset_csv)
  else:
    print('[INFO]', f'Dataset file: "{dataset_csv}" found.')
    
  classifier = ImageClassifier(csv_path=dataset_csv, algorithm=classifier_algorithm, land_marker=land_marker)
  print('[INFO] Opening camera, press "q" to exit..')
  RealTimeEmotionDetector(classifier_model=classifier).recognize()


if __name__ == "__main__":
  """The value of the parameters can change depending on the case."""
  #keras_predict(model, np.zeros((50, 50), dtype=np.uint8))
  run_real_time_emotion_detector(classifier_algorithm='RandomForest',predictor_path='utils/shape_predictor_68_face_landmarks.dat',dataset_csv='data/csv/dataset.csv',dataset_images_dir='data/raw')
  print('Successfully terminated.')