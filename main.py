import os

import keras.losses
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train = X_train.reshape(-1,28*28).astype('float32')/255.0
X_test = X_test.reshape(-1,28*28).astype('float32')/255.0

model = keras.Sequential([
    layers.Dense(512,activation = 'relu'),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10),
])


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.SGD(learning_rate=0.01),
    metrics = ['accuracy'],
)
model.fit(X_train,y_train,epochs = 25,batch_size = 32,verbose = 2)

model.evaluate(X_test,y_test,batch_size = 32,verbose = 2)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs,axis=1)

from sklearn.metrics import  accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy:{accuracy:.4f}')

from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
report = classification_report(y_test,y_pred)
print(f'Classification report:{report}')

cm =confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=range(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()