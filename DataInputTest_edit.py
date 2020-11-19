
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


# NOTE this reads data as 224*224 colored images. Which is compatible with the CNN right now
#  The other one creates grayscale 100*100, which is not compatible with the CNN
#  MobileNetV2 gets 224*244 colored images I don't know why we're using it though :))


# TODO include other data
DIRECTORIES = ["./data/dataset3-artificial",
               "./data/dataset1-real",
               "./data/dataset2-medical"]
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for DIRECTORY in DIRECTORIES:
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = load_img(img_path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)

            data.append(image)
            labels.append(category)

# use one-hot encoding on the labels so we dont have a bunch of strings as are labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)
