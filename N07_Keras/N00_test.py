import tensorflow as tf
from tensorflow import keras
import numpy as np

keras=tf.keras
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)

to_categorical = tf.keras.utils.to_categorical
y_train_enc = np.eye(10)[y_train]
y_test_enc = to_categorical(y_test)

Sequential = keras.Sequential
Activation = keras.layers.Activation
Dense = keras.layers.Dense
Flatten=keras.layers.Flatten

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train_enc, epochs=5, verbose=1)
loss, accuracy = model.evaluate(x_test, y_test_enc, verbose=0)
print("Loss={:.2f}\nAccuracy = {:.2f}".format(loss, accuracy))

