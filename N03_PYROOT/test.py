import tensorflow as tf
import numpy as np
import random
import matplotlib
import pandas as pd
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers


# read data
xy=np.loadtxt('data.csv', delimiter=',', dtype=np.float)
np.random.shuffle(xy)
x_data = xy[:,1:-1]
y_data = xy[:,[-1]]

# training data & test data
tr = int(x_data.shape[0]*0.7)
x_train = x_data[0:tr,:]
y_train = y_data[0:tr,:]
x_test = x_data[tr:,:]
y_test = y_data[tr:,:]

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# HyperParameter
batch_size = 32
training_epochs=10

## --Model
x = layers.Input(shape=(3,))

h = layers.Dense(neu, activation='relu')(x)
#h = layers.Dropout(0.5)(h)

h = layers.Dense(neu, activation='relu')(x)
#h = layers.Dropout(0.5)(h)
h = layers.Dense(neu, activation='relu')(x)

#h = layers.Dropout(0.5)(h)

y = layers.Dense(1, activation='sigmoid')(h)
model = tf.keras.Model(inputs = x,outputs = y)
model.summary()
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)



try:
    model.load_weights(model_weights)
    print('Weights loaded from ' + model_weights)
except IOError:
    print('No pre-trained weights found')
try:
    model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=training_epochs,
        verbose=1,
        callbacks = [
            tf.keras.callbacks.EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(model_weights,
            monitor='val_loss', verbose=True, save_best_only=True)
        ],
        validation_data=(x_test, y_test)
    )
except KeyboardInterrupt:
    print('Training finished early')

model.load_weights(model_weights)
yhat = model.predict(x_test, verbose=1, batch_size=batch_size)
       # score = model.evaluate(images_val, labels_val, sample_weight=weights_val, verbose=1)
       # print 'Validation loss:', score[0]
       # print 'Validation accuracy:', score[1]
np.save(predictions_file, yhat)

test_loss, test_acc = model.evaluate(x_test,y_test)
print('test_acc: ', test_acc)

