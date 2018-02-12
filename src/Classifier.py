import numpy as np
from keras.utils import to_categorical

from keras.preprocessing.image import img_to_array
from PIL import Image

import h5py
import time
import sys

def img2arr(img):
    arr = np.expand_dims(img_to_array(img), axis=0).astype('float32') / 255
    arr.resize(1, 90, 60, 3)
    return arr.copy()

ids_genres = np.genfromtxt('../genres.txt', delimiter=',', dtype=None)

ids = ids_genres[:, 0]
genres = ids_genres[:, 1]

genres_unique = np.unique(genres)

int_map = {}
for i, genre in enumerate(genres_unique):
    int_map[genre] = i

Y = [int_map[genre] for genre in genres]

nClasses = len(genres_unique)
Y_one_hot = to_categorical(Y)

img_size = (60, 90)

X = np.empty((0, 90, 60, 3), dtype='float32')

toolbar_width = 100
progress = 0
sys.stdout.write("[>%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))

for i in range(len(ids)):
    image = Image.open('../posters/'+ids[i].decode('utf-8')+'.jpg')
    resized_image = image.resize(img_size, Image.NEAREST)
    X = np.append(X, img2arr(resized_image), axis=0)

    if progress < i//40:
        progress += 1
        sys.stdout.write("\b")
        sys.stdout.write("=>")
        sys.stdout.flush()

file_data = h5py.File('dataset.h5py', 'w')
file_data.create_dataset('X', data=X)
file_data.create_dataset('Y', data=Y)
file_data.create_dataset('Y_one_hot', data=Y_one_hot)
file_data.create_dataset('ids', data=ids)
file_data.create_dataset('genres', data=genres)
file_data.close()
