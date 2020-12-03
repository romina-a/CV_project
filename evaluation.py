import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from detector import violaJones_face_detection, dnn_face_detection
import os
from sklearn.metrics import classification_report, precision_recall_curve, \
    roc_auc_score, average_precision_score, auc
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

# face detection options (Just to make the code clean)
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


def script():
    C123_path = "./experiments/classifier123/classifier123.model"
    C5_path = "./experiments/classifier5/classifier5.model"
    C_all_path = "./experiments/classifier_all/classifier_all.model"
    lbls, C123_probs = test_model_on_test_images(C123_path)
    lbls, C5_probs = test_model_on_test_images(C5_path)
    lbls, C_all_probs = test_model_on_test_images(C_all_path)

    precision123, recall123, thresholds123 = \
        precision_recall_curve(y_true=lbls.argmax(axis=1), probas_pred=C123_probs[:, 1], pos_label=1)
    precision5, recall5, thresholds5 = \
        precision_recall_curve(y_true=lbls.argmax(axis=1), probas_pred=C5_probs[:, 1], pos_label=1)
    precision_all, recall_all, thresholds_all = \
        precision_recall_curve(y_true=lbls.argmax(axis=1), probas_pred=C_all_probs[:, 1], pos_label=1)

    # finding threshold index
    opt_ind123 = np.where(recall123 >= 0.97)
    opt_ind5 = np.where(recall5 >= 0.97)
    opt_ind_all = np.where(recall_all >= 0.97)

    opt_ind123 = opt_ind123[-1]
    opt_ind5 = opt_ind5[-1]
    opt_ind_all = opt_ind_all[-1]
    # pr curves
    plt.plot(recall123[:-1], precision123[:-1], label="C-123")
    plt.plot(recall_all[:-1], precision_all[:-1], label="C-all")
    plt.plot(recall5[:-1], precision5[:-1], label="C-5")
    plt.plot(recall123[(opt_ind123[-1])], precision123[(opt_ind123[-1])], 'xk')
    plt.plot(recall5[(opt_ind5[-1])], precision5[(opt_ind5[-1])], 'xk')
    plt.plot(recall_all[(opt_ind_all[-1])], precision_all[(opt_ind_all[-1])], 'xk')
    plt.legend()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("./prcurves.png")
    plt.show()

    # individual precision and recall against threshold
    plt.plot(thresholds123, precision123[:-1], label="precision")
    plt.plot(thresholds123, recall123[:-1], label="recall")
    plt.plot(thresholds123[opt_ind123[-1]], precision123[opt_ind123[-1]], 'xk')
    plt.plot(thresholds123[opt_ind123[-1]], recall123[opt_ind123[-1]], 'xk')
    plt.xlabel("Threshold")
    plt.title("C-123 precision and recall curve")
    plt.legend()
    plt.savefig("./p&rcurve_c_123.png")
    plt.show()

    plt.plot(thresholds5, precision5[:-1], label="precision")
    plt.plot(thresholds5, recall5[:-1], label="recall")
    plt.plot(thresholds5[opt_ind5[-1]], precision5[opt_ind5[-1]], 'xk')
    plt.plot(thresholds5[opt_ind5[-1]], recall5[opt_ind5[-1]], 'xk')
    plt.xlabel("Threshold")
    plt.legend()
    plt.title("C-5 precision and recall curve")
    plt.savefig("./p&rcurve_c_5.png")
    plt.show()

    plt.plot(thresholds_all, precision_all[:-1], label="precision")
    plt.plot(thresholds_all, recall_all[:-1], label="recall")
    plt.plot(thresholds_all[opt_ind_all[-1]], precision_all[opt_ind_all[-1]], 'xk')
    plt.plot(thresholds_all[opt_ind_all[-1]], recall_all[opt_ind_all[-1]], 'xk')
    plt.xlabel("Threshold")
    plt.legend()
    plt.title("C-all precision and recall curve")
    plt.savefig("./p&rcurve_c_all.png")
    plt.show()


    print("AP C-123: {}"
          .format(average_precision_score(y_true=lbls.argmax(axis=1),
                                          y_score=C123_probs[:, 1],
                                          pos_label=1)))
    print("AP C-5: {}"
          .format(average_precision_score(y_true=lbls.argmax(axis=1),
                                          y_score=C5_probs[:, 1],
                                          pos_label=1)))
    print("AP C-all: {}"
          .format(average_precision_score(y_true=lbls.argmax(axis=1),
                                          y_score=C_all_probs[:, 1],
                                          pos_label=1)))
    print("AUC-ROC C-123: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C123_probs[:, 1])))
    print("AUC-ROC C-5: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C5_probs[:, 1])))
    print("AUC-ROC C-all: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C_all_probs[:, 1])))
    print("AUC-PR C-123: {}"
          .format(auc(recall123, precision123)))
    print("AUC-PR C-5: {}"
          .format(auc(recall5, precision5)))
    print("AUC_PR C-all: {}"
          .format(auc(recall_all, precision_all)))

script()