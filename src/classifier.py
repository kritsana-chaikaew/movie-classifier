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
from keras.utils import to_categorical
import os
from keras.optimizers import SGD, Adam

dataset_name = str(input('Dataset Name: '))
augmentation = str(input('Data Augmentation [y/N]: '))

with h5py.File('../datasets/'+dataset_name+'/'+dataset_name+'.h5py', 'r') as file_data:
    X = file_data['X'][:].astype('float32') / 255
    Y = file_data['Y'][:]
with open('../datasets/classes.txt') as file_classes:
    classes = np.loadtxt(file_classes, dtype='U', usecols=(0,))
with open('../datasets/'+dataset_name+'/ids.txt') as file_ids:
    ids = np.loadtxt(file_ids, dtype='U', usecols=(0,))

print('Data shape: ', X.shape)

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15)

train_X, valid_X, train_Y, valid_Y = \
        train_test_split(train_X, train_Y, test_size=0.2)

batch_size = int(input('Batch size [64]: ') or 64)
epochs = int(input('Epochs [20]: ') or 20)
num_classes = len(classes)
input_shape = X[0].shape

train_label = to_categorical(train_Y, num_classes)
test_label = to_categorical(test_Y, num_classes)
valid_label = to_categorical(valid_Y, num_classes)

movie_model = Sequential()
movie_model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation='linear',
        padding='same',
        input_shape=input_shape))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D((2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(256, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Conv2D(512, (3, 3), activation='linear',padding='same'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
movie_model.add(Dropout(0.2))

movie_model.add(Flatten())
movie_model.add(Dense(128, activation='linear'))
movie_model.add(LeakyReLU(alpha=0.1))
movie_model.add(Dropout(0.5))

movie_model.add(Dense(num_classes, activation='softmax'))

movie_model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=Adam(),
        metrics=['accuracy'])

movie_model.summary()

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

test_eval = movie_model.evaluate(test_X, test_label, verbose=1)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

accuracy = movie_train.history['acc']
val_accuracy = movie_train.history['val_acc']
loss = movie_train.history['loss']
val_loss = movie_train.history['val_loss']
epochs = range(len(accuracy))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

predicted_classes = movie_model.predict_classes(test_X)

correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[correct])
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
    plt.tight_layout()
plt.show()

incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(test_X[incorrect])
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
    plt.tight_layout()
plt.show()

if not os.path.exists("../models"):
    os.makedirs("../models")

if str(input('Save model? [y/n]: ')) == 'y':
    save_name = str(input('Save name: '))
    movie_model.save('../models/'+save_name+'.h5py')
