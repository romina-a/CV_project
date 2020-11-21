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
    # change the output to show x1,y1,x2,y2 instead of x1,y1,w,h
    faces[:, 2] = faces[:, 2] + faces[:, 0]
    faces[:, 3] = faces[:, 3] + faces[:, 1]
    return faces


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

    confidence = 0.5  # can change confidence to get more/less likely faces
    (w, h) = image.shape[0:2]
    # using np indexing to get only the location of the faces for
    # only the rows that have confidence more than confidence
    faces = detections[0, 0, detections[0, 0, :, 2] > confidence, 3:7] * np.array([h, w, h, w])
    # convert the locations to int and return
    return faces.astype(int)


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

    # getting ViolaJones faces and adding rectangles
    VJ_faces = violaJones_face_detection(image)
    VJ = image.copy()
    for x, y, w, h in VJ_faces:
        cv2.rectangle(VJ, (x, y), (w, h), (0, 0, 0), 4)

    # getting dnn faces and adding rectangles
    dnn_faces = dnn_face_detection(image)
    dnn = image.copy()
    for x1, y1, x2, y2 in dnn_faces:
        print("is {},{},{},{}".format(x1, y1, x2, y2))
        cv2.rectangle(dnn, (x1, y1), (x2, y2), (0, 0, 0), 4)

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


if __name__ == "__main__":
    test()
