import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout




(X_train,y_train),(X_test,y_test) = cifar10.load_data()
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

model = keras.Sequential([
     keras.Input(shape=(32,32,3)),
     layers.Conv2D(32,3,padding ='valid',activation = 'relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     layers.MaxPooling2D(pool_size = (2,2)),
     layers.Conv2D(64, 3, padding='valid', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Conv2D(128, 3, padding='valid', activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001)),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Flatten(),
     layers.Dropout(0.5),
     layers.Dense(64,activation = 'relu'),
     layers.Dense(10),
])
# model = tf.keras.Sequential([
#     Conv2D(32, (3,3), activation='relu',
#            kernel_regularizer=tf.keras.regularizers.l2(0.001),
#            input_shape=(32,32,3)),
#     BatchNormalization(),
#     Dropout(0.3),
#
#     Conv2D(64, (3,3), activation='relu',
#            kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.3),
#
#     tf.keras.layers.Flatten(),
#     layers.Dense(128, activation='relu',
#           kernel_regularizer=tf.keras.regularizers.l2(0.001)),
#     Dropout(0.5),
#
#     layers.Dense(10)
# ])

#model.summary()
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer= keras.optimizers.SGD(learning_rate=0.01),
    metrics = ['accuracy'],
)
model.fit(X_train,y_train,epochs = 150,batch_size = 32,verbose = 2)
model.evaluate(X_test,y_test,batch_size = 32,verbose = 2)