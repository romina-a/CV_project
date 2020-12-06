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
DEFAULT_TEST_DIRECTORIES = ["data/test/test_images/",
                            "data/test/test_images2/",
                            "data/test/jean_louis_mask_dataset/",
                            "data/test/Pamir_Amiry_Data_Set/",
                            "data/test/Aggrim_Arora_Data/",
                            "data/test/christian_augustyn_data",
                            "data/test/martin_dataset",
                            ]


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
    print("C-123 with threshold 0.5")
    lbls, C123_probs = test_model_on_test_images(C123_path)
    print("C-5 with threshold 0.5")
    lbls, C5_probs = test_model_on_test_images(C5_path)
    print("C-all with threshold 0.5")
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

    o123 = (opt_ind123[0][-1])
    o5 = (opt_ind5[0][-1])
    o_all = (opt_ind_all[0][-1])

    print("optimal thresholds are:\nC-123:{}\nC-5:{}\nC-all:{}\n"
          .format(thresholds123[o123],
                  thresholds5[o5],
                  thresholds_all[o_all])
          )
    print("optimal recalls are:\nC-123:{}\nC-5:{}\nC-all:{}\n"
          .format(recall123[o123],
                  recall5[o5],
                  recall_all[o_all])
          )
    print("optimal precision are:\nC-123:{}\nC-5:{}\nC-all:{}\n"
          .format(precision123[o123],
                  precision5[o5],
                  precision_all[o_all])
          )
    print("optimal f1 are:\nC-123:{}\nC-5:{}\nC-all:{}\n"
          .format(2*recall123[o123]*precision123[o123]/(recall123[o123]+precision123[o123]),
                  2*recall5[o5]*precision5[o5]/(recall5[o5]+precision5[o5]),
                  2*recall_all[o_all]*precision_all[o_all]/(recall_all[o_all]+precision_all[o_all]))
          )

    # pr curves
    fig, ax = plt.subplots()
    ax.plot(recall123[:-1], precision123[:-1], label="C-123", color='c')
    ax.plot(recall5[:-1], precision5[:-1], label="C-5", color='m')
    ax.plot(recall_all[:-1], precision_all[:-1], label="C-all", color='y')
    ax.plot(recall123[o123], precision123[o123], 'xc', label="C-123 optimal point")
    ax.plot(recall5[o5], precision5[o5], 'xm', label="C-5 optimal point")
    ax.plot(recall_all[o_all], precision_all[o_all], 'xy', label="C-all optimal point")
    ax.axvline(0.97, linestyle='--', color='k', linewidth=0.5)
    # ax.set_xticks(list(ax.get_xticks()[:-1]) + [0.97])
    ax.legend()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.savefig("./prcurves.png")
    fig.show()

    # individual precision and recall against threshold
    fig, ax = plt.subplots()
    ax.plot(thresholds123, precision123[:-1], label="precision")
    ax.plot(thresholds123, recall123[:-1], label="recall")
    ax.plot(thresholds123[o123], precision123[o123], 'xk')
    ax.plot(thresholds123[o123], recall123[o123], 'xk')
    ax.set_xlabel("Threshold")
    ax.legend()
    ax.set_title("C-123 precision and recall curve")
    ax.axhline(0.97, linestyle='--', color='k', linewidth=0.5)
    # ax.set_yticks(list(ax.get_yticks()[:-1]) + [0.97])
    fig.savefig("./p&rcurve_c_123.png")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(thresholds5, precision5[:-1], label="precision")
    ax.plot(thresholds5, recall5[:-1], label="recall")
    ax.plot(thresholds5[o5], precision5[o5], 'xk')
    ax.plot(thresholds5[o5], recall5[o5], 'xk')
    ax.set_xlabel("Threshold")
    ax.legend()
    ax.set_title("C-5 precision and recall curve")
    ax.axhline(0.97, linestyle='--', color='k', linewidth=0.5)
    # ax.set_yticks(list(ax.get_yticks()[:-1]) + [0.97])
    fig.savefig("./p&rcurve_c_5.png")
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(thresholds_all, precision_all[:-1], label="precision")
    ax.plot(thresholds_all, recall_all[:-1], label="recall")
    ax.plot(thresholds_all[o_all], precision_all[o_all], 'xk')
    ax.plot(thresholds_all[o_all], recall_all[o_all], 'xk')
    ax.set_xlabel("Threshold")
    ax.legend()
    ax.set_title("C-all precision and recall curve")
    ax.axhline(0.97, linestyle='--', color='k', linewidth=0.5)
    # ax.set_yticks(list(ax.get_yticks())[:-1] + [0.97])
    fig.savefig("./p&rcurve_c_all.png")
    fig.show()

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
    print()
    print("AUC-ROC C-123: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C123_probs[:, 1])))
    print("AUC-ROC C-5: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C5_probs[:, 1])))
    print("AUC-ROC C-all: {}"
          .format(roc_auc_score(y_true=lbls.argmax(axis=1),
                                y_score=C_all_probs[:, 1])))
    print()
    print("AUC-PR C-123: {}"
          .format(auc(recall123, precision123)))
    print("AUC-PR C-5: {}"
          .format(auc(recall5, precision5)))
    print("AUC_PR C-all: {}"
          .format(auc(recall_all, precision_all)))
    print()


script()
