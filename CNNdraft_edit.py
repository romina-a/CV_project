from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report
import numpy as np
from matplotlib import pyplot as plt

INIT_LR = 1e-4  # initial learning rate
EPOCHS = 20  #
BS = 32  # Batch size

# ---------------------------------------------------------------
# Elie's stuff with some comment added to that. Didn't change his.

# Splitting data
(trainX, testX, trainY, testY) = \
    train_test_split(data, labels,
                     test_size=0.20, stratify=labels,
                     random_state=42)  # ask data guys to change target to lable makes more sense when reading though

# used this for data augmentation while training
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,  # sort of like affine
    horizontal_flip=True,
    fill_mode="nearest"  # how to fill points outside the image boundaries
)

# We are using a pretrained model but I don't get why.
# It has something to do with fine tuning the network (initialize weights?)
# Anyways this model will not be trained
# TODO I don't get what input_tensor does maybe remove that and see what happens
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,# because we're adding our own head to the model
                        input_tensor=Input(shape=(224, 224, 3)),
                        input_shape=(224, 224, 3)
                        )
# input is BS*224*224*3 output is input is BS*7*7*1280

# adding a DNN on top of the baseModes. This is what we train
# base
# -->
# AveragePooling (Shrinks data by taking average in each 7*7 box (so data size with be 1/7 in x and y)
# (it'll become BS*1*1*1280))
# -->
# Flatten (creates a 1d array) # TODO don't know what these layers do
# -->
# Dense (a fully connected layer?)
# -->
# dropout (used for preventing overfitting)
# -->
# softmax for classification with two classes
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# don't train the mobileNetV2
for layer in baseModel.layers:
    layer.trainable = False

# compile model, get ready to train
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# training (after training we have the classifier. We just need to give it data)
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# testing
# getting the predicted probabilities
predIdxs = model.predict(testX, batch_size=BS)

# printing report
print(classification_report(testY.argmax(axis=1), predIdxs.argmax(axis=1)))

# save the model. # TODO see what happens if change the format to tf
model.save('detector.model', save_format="h5")

# plot training history
# looks like 10 Epochs is enough for our model
# TODO why val_accuracy is so high in the beginning
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig("./plots")



