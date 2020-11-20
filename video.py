from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


net = cv2.dnn.readNetFromCaffe("./dnn/deploy.prototxt", "./dnn/weights.caffemodel")

face_detector = cv2.CascadeClassifier("./frontalFace10/haarcascade_frontalface_default.xml")
# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

model = load_model("./detector.model")

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detecting the face with viola jones
    faces = face_detector.detectMultiScale(frame,
                                           scaleFactor=None,
                                           minNeighbors=None,
                                           flags=None,
                                           minSize=None,
                                           maxSize=None
                                           )
    # loop over the detections
    for x, y, w, h in faces:
        face = frame[y:y + h, x:x + w, :]
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        predIdxs = model.predict(face)[0]

        if predIdxs.argmax() == 0:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
