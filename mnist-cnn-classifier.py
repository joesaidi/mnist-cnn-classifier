import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model,Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

train_ds, test_ds = tfds.load(
    "mnist",
    split=["train", "test"],
    as_supervised=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Model
inputs = Input(shape=(28, 28, 1))
x = layers.Conv2D(128, (3, 3), activation="relu")(inputs)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(32, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)

outputs = layers.Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=test_ds, epochs=10)
model.evaluate(test_ds)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)



plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.colorbar()
plt.show(block=True)

