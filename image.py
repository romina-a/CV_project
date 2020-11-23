import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from detector import violaJones_face_detection, dnn_face_detection
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


# face detection options (Just to make to code clean)
fd_options = {
    'ViolaJones': violaJones_face_detection,
    'DNN': dnn_face_detection
}
# default classifier model and detection method
DEFAULT_CLASSIFIER_PATH = "experiments/classifier3/classifier3.model"
DEFAULT_FACEDETECTION_METHOD = "ViolaJones"
# default image
DEFAUL_IMAGE_PATH = "data/test_images/with_mask/1.jpg"

DEFAULT_TEST_DIRECTORIES = ["data/test_images/"]


# THIS FUNCTION IS TO EVALUATE THE MODELS AND COMPARE THEM
# for all images in the test directories,
# detects faces and labels as mask or no mask with the provided model
# then compares classifier output with the real labels and prints success rate
def test_model_on_test_images(classifier_path=None, fd_method=None, test_directories=None):
    categories = ["with_mask", "without_mask"]

    # set directory of the test data
    if test_directories is not None:
            test_dir = test_directories
    else:
            test_dir = DEFAULT_TEST_DIRECTORIES

    # set the face detector function
    if fd_method is None:
        face_detect = fd_options[DEFAULT_FACEDETECTION_METHOD]
    else:
        face_detect = fd_options[fd_method]

    # Load the model
    if classifier_path is None:
        model = load_model(DEFAULT_CLASSIFIER_PATH)
    else:
        model = load_model(classifier_path)

    # real labels
    labels = []
    # predicted labels
    pred_labels = []
    # loop through all the images in the test folder detect and label faces and save predicted labels and real ones
    for directory in test_dir:
        for category in categories:
            path = os.path.join(directory, category)
            for img in os.listdir(path):

                img_path = os.path.join(path, img)
                try:
                    image = load_img(img_path)  # load the image
                except:
                    continue
                image = img_to_array(image)
                faces = face_detect(image)
                for x1, y1, x2, y2 in faces:
                    # crop the detected face from the frame and preprocess the image
                    face = image[y1:y2, x1:x2, :]
                    if face.shape[0] == 0 or face.shape[1] == 0:
                        continue
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face) # preprocessing the data for the mobileNetV2 CNN
                    face = np.expand_dims(face, axis=0)

                    # predicted label
                    predIdxs = model.predict(face)[0]
                    pred_labels.append(predIdxs)
                    # real label
                    labels.append(category)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    # real and predicted label
    labels = np.array(labels)
    pred_labels = np.array(pred_labels)
    print(classification_report(labels.argmax(axis=1),pred_labels.argmax(axis=1)))


# THIS FUNCTION GETS AN IMAGE AND MARKS THE FACES AS MASK OR NO MASK AND SHOWS THE RESULT
def detect_and_mark(classifier_path=None, fd_method=None, image_path=None, save_path=None):
    """

    :param classifier_path: path to the classifier model
    :param fd_method: 'ViolaJones' or 'DNN' default is ViolaJones
    :param image_path: path to the image to be marked
    :param save_path: path to save the marked image
    """

    # Load the model
    if classifier_path is None:
        model = load_model(DEFAULT_CLASSIFIER_PATH)
    else:
        model = load_model(classifier_path)

    # set the face detector function
    if fd_method is None:
        face_detect = fd_options[DEFAULT_FACEDETECTION_METHOD]
    else:
        face_detect = fd_options[fd_method]

    # set the path to the image
    if image_path is None:
        image_path = DEFAUL_IMAGE_PATH

    # read the image
    image = cv2.imread(image_path).copy()

    # detect the faces
    faces = face_detect(image)

    # loop over the detections and annotate the image
    for x1, y1, x2, y2 in faces:
        # crop the detected face from the frame
        face = image[y1:y2, x1:x2, :]
        # dnn sometimes gives faces with 0 dimensions to prevent error:
        if face.shape[0] == 0 or face.shape[1] == 0:
            continue
        # preprocessing the face for the mask classifier
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # pass the face to the mask classifier and get the label
        predIdxs = model.predict(face)[0]

        # add the frames around the faces based on the prediction
        if predIdxs.argmax() == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # show the annotated image
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # if save_path provided save the annotated image
    if save_path is not None:
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-cp", "--classifier_path", required=False, default=DEFAULT_CLASSIFIER_PATH,
                    help="path to the classifier model")
    ap.add_argument("-fd", "--face_detection_method", required=False, default=DEFAULT_FACEDETECTION_METHOD,
                    help="face detection method 'DNN' or 'ViolaJones'")
    ap.add_argument("-sp", "--save_path", required=False, default=None,
                    help="path to save the result")
    ap.add_argument("-im", "--image_path", required=False, default=DEFAUL_IMAGE_PATH,
                    help="path to the image")
    args = vars(ap.parse_args())
    print("face_detection_method is : {}".format(args['face_detection_method']))
    print("classifier is : {}".format(args['classifier_path']))

    detect_and_mark(classifier_path=args['classifier_path'],
                    fd_method=args['face_detection_method'],
                    image_path=args['image_path'],
                    save_path=args['save_path'])
