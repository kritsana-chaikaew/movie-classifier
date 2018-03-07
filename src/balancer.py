import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

with h5py.File('../datasets/100/100.h5py', 'r') as file_data:
    X = file_data['X'][:].astype('float32')
    Y = file_data['Y'][:]

data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

counts = np.unique(Y, return_counts=True)[1]

diffs = counts - max(counts)
print(diffs)
