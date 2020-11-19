from cv2 import CascadeClassifier as CC
from cv2 import imread, imshow
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

img_path = "./data/test_images/1.jpg"

face_detector = CC("haarcascade_frontalface_default.xml")

image = imread(img_path)


face_detector.detectMultiScale(image)

