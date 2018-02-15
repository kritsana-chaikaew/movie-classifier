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

from keras.preprocessing.image import ImageDataGenerator

dataset_name = str(input('Dataset Name: '))
augmentation = str(input('Data Augmentation [y/N]: '))

with h5py.File(dataset_name+'.h5py', 'r') as file_data:
    X = file_data['X'][:]
    Y = file_data['Y'][:]
    Y_one_hot = file_data['Y_one_hot'][:]
    classes = file_data['genres_unique'][:]

print('Data shape: ', X.shape)

train_X, test_X, train_Y_one_hot, test_Y_one_hot = \
        train_test_split(X, Y_one_hot, test_size=0.15, random_state=13)

train_X, valid_X, train_label, valid_label = \
        train_test_split(
                train_X, train_Y_one_hot,
                test_size=0.2, random_state=13)

batch_size = 1
epochs = 20
num_classes = len(classes)

movie_model = Sequential()
movie_model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation='linear',
        padding='same',
        input_shape=(X.shape[1], X.shape[2], 3)))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D((2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.3))

movie_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.3))

movie_model.add(Flatten())
movie_model.add(Dense(128, activation='linear'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(Dropout(0.3))
movie_model.add(Dense(num_classes, activation='softmax'))

movie_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

if augmentation == 'y' or augmentation == 'Y':
    print('using data augmentation')
    datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

    movie_train = movie_model.fit_generator(
            datagen.flow(train_X, train_label, batch_size=batch_size),
            steps_per_epoch=len(train_X)//batch_size,
            epochs=epochs, verbose=1,
            validation_data=datagen.flow(
                    valid_X, valid_label,
                    batch_size=batch_size),
            validation_steps=len(valid_X)//batch_size)
else:
    movie_train = movie_model.fit(
            train_X, train_label,
            batch_size=batch_size,
            epochs=epochs,verbose=1,
            validation_data=(valid_X, valid_label))

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
