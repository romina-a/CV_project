import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from detector import violaJones_face_detection, dnn_face_detection
import os
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# face detection options (Just to make to code clean)
fd_options = {
    'ViolaJones': violaJones_face_detection,
    'DNN': dnn_face_detection
}
# default classifier model and detection method
DEFAULT_CLASSIFIER_PATH = "experiments/classifier3/classifier3.model"
DEFAULT_TEST_DIRECTORIES = ["data/test_images/", "data/test_images2/"]


# THIS FUNCTION IS TO EVALUATE THE MODELS AND COMPARE THEM
# for all images in the test directories,
# detects faces and labels as mask or no mask with the provided model
# then compares classifier output with the real labels and prints success rate
def test_model_on_test_images(classifier_path=None, fd_method='DNN', test_directories=None,
                              PR_curve=False, threshold=0.5):
    categories = ["with_mask", "without_mask"]

    # set directory of the test data
    if test_directories is not None:
        test_dir = test_directories
    else:
        test_dir = DEFAULT_TEST_DIRECTORIES

    # set the face detector function
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
                    face = preprocess_input(face)  # preprocessing the data for the mobileNetV2 CNN
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
    pred_labels_t = np.zeros(pred_labels.shape[0])
    pred_labels_t[pred_labels[:, 1] > threshold] = 1
    print(classification_report(labels.argmax(axis=1), pred_labels_t))
    # for example for label 0:
    # precision: the number of correcly reported 0s/ the number of all reported 0s
    #   (what percent of algorhtm's 0s were real 0s)
    # recall: the number of correcly reported 0s/ the number of all 0s
    #   (what percent of data's 0s the algorithm caught)
    if PR_curve:
        precision, recall, thresholds = \
            precision_recall_curve(y_true=labels.argmax(axis=1), probas_pred=pred_labels[:, 0], pos_label=0)
        plt.plot(recall, precision)
        plt.show()
        plt.grid

    return labels, pred_labels
