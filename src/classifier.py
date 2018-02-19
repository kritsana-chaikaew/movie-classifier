import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import os

dataset_name = str(input('Dataset Name: '))
with h5py.File('../datasets/'+dataset_name+'/'+dataset_name+'.h5py', 'r') as dataset:
    X = dataset['X'][:] / 255
    Y = dataset['Y'][:]
with open('../datasets/classes.txt') as file_classes:
    classes = np.loadtxt(file_classes, dtype='U', usecols=(0,))
    num_classes = len(classes)

print('Data shape: ', X.shape)
plt.imshow(X[0])
plt.title('Class %d' % Y[0])
plt.show()

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2)

batch_size = int(input('Batch size [64]: ') or 64)
epochs = int(input('Epochs [20]: ') or 20)
input_shape = X[0].shape

train_label = to_categorical(train_Y, num_classes)
valid_label = to_categorical(valid_Y, num_classes)

model = Sequential()
model.add(Conv2D(
        32,
        kernel_size=(3, 3),
        activation='linear',
        padding='same',
        input_shape=input_shape))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))

model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])

movie_train = model.fit(
        train_X, train_label,
        batch_size=batch_size,
        epochs=epochs,verbose=1,
        validation_data=(valid_X, valid_label))

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

predicted_classes = model.predict(valid_X)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

correct = np.where(predicted_classes == valid_Y)[0]
print("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(valid_X[correct])
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], valid_Y[correct]))
    plt.tight_layout()
plt.show()

incorrect = np.where(predicted_classes != valid_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(valid_X[incorrect])
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], valid_Y[incorrect]))
    plt.tight_layout()
plt.show()

if not os.path.exists("../models"):
    os.makedirs("../models")

if str(input('Save model? [y/n]: ')) == 'y':
    model_name = str(input('Save name: '))
    model.save('../models/'+model_name+'.h5py')
