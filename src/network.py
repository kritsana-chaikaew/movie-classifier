import numpy as np
import h5py
from sklearn.model_selection import train_test_split

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
