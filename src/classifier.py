import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

import os
from keras.optimizers import SGD
from keras import regularizers

def construct_model ():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='tanh', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='tanh'))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='tanh'))
    model.add(Conv2D(64, (3, 3), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

dataset_name = str(input('Dataset Name: '))
with h5py.File('../datasets/'+dataset_name+'/'+dataset_name+'.h5py', 'r') as file_data:
    X = file_data['X'][:]
    Y = file_data['Y'][:]
with open('../datasets/classes.txt') as file_classes:
    classes = np.loadtxt(file_classes, dtype='U', usecols=(0,))
    num_classes = len(classes)
with open('../datasets/'+dataset_name+'/ids.txt') as file_ids:
    ids = np.loadtxt(file_ids, dtype='U', usecols=(0,))

for x in X:
    x = x.astype('float32') / 255

train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2)

if str(input('Load model? [y/n]: ')) == 'y':
    model_name = str(input('Model name: '))
    model = load_model('../models/'+model_name+'.h5py')
else:
    batch_size = int(input('Batch size [64]: ') or 64)
    epochs = int(input('Epochs [20]: ') or 20)
    input_shape = X[0].shape

    train_label = to_categorical(train_Y, num_classes)
    valid_label = to_categorical(valid_Y, num_classes)

    model = construct_model()

    model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=SGD(lr=0.01, momentum=0.2, decay=0.0, nesterov=False),
            metrics=['accuracy'])
    model.summary()

    train = model.fit(
            train_X, train_label,
            batch_size=batch_size,
            epochs=epochs,verbose=1,
            validation_data=(valid_X, valid_label))

    accuracy = train.history['acc']
    val_accuracy = train.history['val_acc']
    loss = train.history['loss']
    val_loss = train.history['val_loss']
    epochs = range(len(accuracy))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, accuracy, 'r', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    if not os.path.exists("../models"):
        os.makedirs("../models")

    if str(input('Save model? [y/n]: ')) == 'y':
        model_name = str(input('Model name: '))
        model.save('../models/'+model_name+'.h5py')

predicted_classes = model.predict_classes(valid_X)

correct = np.where(predicted_classes == valid_Y)[0]
incorrect = np.where(predicted_classes != valid_Y)[0]
print("Found %d correct labels" % len(correct))
print("Found %d incorrect labels" % len(incorrect))
print("Accuracy", len(correct) / (len(correct) + len(incorrect)))

for i, correct in enumerate(correct[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(valid_X[correct])
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], valid_Y[correct]))
    plt.tight_layout()
plt.show()

for i, incorrect in enumerate(incorrect[:9]):
    plt.subplot(3, 3, i+1)
    plt.imshow(valid_X[incorrect])
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], valid_Y[incorrect]))
    plt.tight_layout()
plt.show()
