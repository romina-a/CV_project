import cv2
import numpy as np
import matplotlib.pyplot as plt

default_img_path = "./data/test_images/1.jpg"


# This is to compare HaarCascade (ViolaJones), with a pretrained Deep Neural Net
# for face detection
# each function gets an image and adds rectangles around the detected faces.

# ---------------------------------------------------------------------------
# detecting the face with viola jones
def violaJones_face_detection(image):
    # Loading the model
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
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 4)

    return image


# ------------------------------------------------------------------------------
# detecting the face with dnn
def dnn_face_detection(image):
    # loading the model
    net = cv2.dnn.readNetFromCaffe("./dnn/deploy.prototxt", "./dnn/weights.caffemodel")
    # creating a blob: preprocess the image. also make the image size compatible with the dnn
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass blob to dnn and detect
    net.setInput(blob)
    detections = net.forward()

    # add rectangles
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        (w, h) = image.shape[0:2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([h, w, h, w])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 0), 4)

    return image


def test(img_path=None, gray=False):
    """
    :param img_path: path to the image
    :param gray: convert to gray before processing default is False
    """
    if img_path is None:
        img_path = default_img_path
    # reading the image
    image = cv2.imread(img_path).copy()
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    VJ = violaJones_face_detection(image.copy())
    dnn = dnn_face_detection(image.copy())
    # plotting two images
    plt.axis("off")
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(VJ, cv2.COLOR_BGR2RGB))
    plt.title("ViolaJones")
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(dnn, cv2.COLOR_BGR2RGB))
    plt.title("DNN")
    plt.show()
    # cv2.imwrite("sampleoutput/dnn/dnn50confidence.jpg", image)
