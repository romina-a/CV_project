import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

img_path = "./data/test_images/1.jpg"

# read the image
image = cv2.imread(img_path).copy()
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ---------------------------------------------------------------------------
model = load_model("detector.model")
# ---------------------------------------------------------------------------
# detecting the face with viola jones
face_detector = cv2.CascadeClassifier("./frontalFace10/haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(image,
                                       scaleFactor=None,
                                       minNeighbors=None,
                                       flags=None,
                                       minSize=None,
                                       maxSize=None
                                       )

# adding the detected faces to the image
for x, y, w, h in faces:
    face = image[y:y+h, x:x+w, :]
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    predIdxs = model.predict(face)[0]

    if predIdxs.argmax() == 0:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

plt.imshow(image)
plt.show()

# saving image with detected faces
cv2.imwrite("sampleoutput/dnn/dnn50confidence.jpg", image)

# ------------------------------------------------------------------------------
# detecting the face with dnn

# loading the neural net from file
net = cv2.dnn.readNetFromCaffe("./dnn/deploy.prototxt", "./dnn/weights.caffemodel")
# creating a blob: preprocess the image. also make the image size compatible with the dnn
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass blob to dnn
net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    (w, h) = image.shape[0:2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
