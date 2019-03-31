from __future__ import print_function
import pickle
from tensorflow import keras

batch_size = 64
number_of_categories = 10
epochs = 25

with open('train.pickle', 'rb') as handle:
    x_train, y_train = pickle.load(handle)
with open('test.pickle', 'rb') as handle:
    x_test = pickle.load(handle)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(64, (5, 5), input_shape=x_train.shape[1:], strides=1))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(keras.layers.Conv2D(64, (5, 5), strides=1))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(384))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(192))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dense(10))
model.add(keras.layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train, number_of_categories)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True)

predictions = model.predict(x_test)
