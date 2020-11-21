from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from detector import violaJones_face_detection, dnn_face_detection
import argparse

# face detection options (Just to make to code clean)
fd_options = {
    'ViolaJones': violaJones_face_detection,
    'DNN': dnn_face_detection
}
# default classifier model and detection method
default_classifier_path = "./classifier.model"
default_faceDetection_method = "ViolaJones"


def start_video(classifier_path=None, fd_method=None):
    """

    :param classifier_path: path to the classifier model
    :param fd_method: 'ViolaJones' or 'DNN' default is ViolaJones
    """

    # Load the model
    if classifier_path is None:
        model = load_model(default_classifier_path)
    else:
        model = load_model(classifier_path)

    # set the face detector function
    if fd_method is None:
        face_detect = fd_options[default_faceDetection_method]
    else:
        face_detect = fd_options[fd_method]

    # initialize the video stream and allow the camera sensor to warmup
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detecting the face
        faces = face_detect(frame)

        # loop over the detections
        for x1, y1, x2, y2 in faces:
            # crop the detected face from the frame and preprocess the image
            face = frame[y1:y2, x1:x2, :]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass that to the mask detector
            predIdxs = model.predict(face)[0]

            # add the frames around the faces
            if predIdxs.argmax() == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-cp", "--classifier_path", required=False, default=default_classifier_path,
                    help="path to the classifier models")
    ap.add_argument("-fd", "--face_detection_method", required=False, default=default_faceDetection_method,
                    help="path to the classifier models")
    args = vars(ap.parse_args())
    print("face_detection_method is : {}".format(args['face_detection_method']))
    print("classifier is : {}".format(args['classifier_path']))

    start_video(classifier_path=args['classifier_path'], fd_method=args['face_detection_method'])
