import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

file_data = h5py.File('dataset.h5py', 'r')
X = file_data['X'][:]
Y = file_data['Y'][:]
Y_one_hot = file_data['Y_one_hot'][:]
classes = file_data['genres_unique'][:]
file_data.close()

train_X, test_X, train_Y_one_hot, test_Y_one_hot = train_test_split(X, Y_one_hot, test_size=0.15, random_state=13)
print(train_X.shape, test_X.shape, train_Y_one_hot.shape, test_Y_one_hot.shape)

train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

batch_size = 64
epochs = 100
num_classes = 21

movie_model = Sequential()
movie_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(90,60,3)))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D((2, 2),padding='same'))
movie_model.add(Dropout(0.1))

movie_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.3))

movie_model.add(Flatten())
movie_model.add(Dense(128, activation='linear'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(Dropout(0.4))
movie_model.add(Dense(num_classes, activation='softmax'))

movie_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.000001),metrics=['accuracy'])

movie_train = movie_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

test_eval = movie_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = movie_train.history['acc']
val_accuracy = movie_train.history['val_acc']
loss = movie_train.history['loss']
val_loss = movie_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
