import numpy as np
from keras.utils import to_categorical

from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

import h5py
import sys
import glob

def img2arr(img):
    arr = np.expand_dims(img_to_array(img), axis=0).astype('float32') / 255
    arr.resize(1, 90, 60, 3)
    return arr.copy()

file_name = str(input('File name: '));

ids_genres = np.empty((0, 2), dtype=None)
genres = np.empty((0,), dtype=None)

pattern = '../' + ('[0-9]' * 6 + '-' + '[0-9]' * 6 + '.txt')
files = glob.glob(pattern)

if len(files) == 0:
    print('Download data first')
    exit()

for filename in files:
    array = np.genfromtxt(filename, delimiter=',', dtype=None)
    if array.ndim == 1:
        array = [array]
    ids_genres = np.append(ids_genres, array, axis=0)

ids = ids_genres[:, 0]
genre_sets = ids_genres[:, 1]

# one genre
for genre_set in genre_sets:
    genre = genre_set.decode('utf-8').split(':')[0].encode('utf-8')
    genres = np.append(genres, [genre], axis=0)

genres_unique = np.unique(genres)

int_map = {}
for i, genre in enumerate(genres_unique):
    int_map[genre] = i

Y = [int_map[genre] for genre in genres]

Y_one_hot = to_categorical(Y)

img_size = (60, 90)

X = np.empty((0, 90, 60, 3), dtype='float32')

for i in range(len(ids)):
    image = Image.open('../posters/'+ids[i].decode('utf-8')+'.jpg')
    resized_image = image.resize(img_size, Image.NEAREST)
    X = np.append(X, img2arr(resized_image), axis=0)

    sys.stdout.write("\b"*10)
    sys.stdout.flush()
    sys.stdout.write("{}\t%".format(i/40))
    sys.stdout.flush()

file_data = h5py.File(file_name+'.h5py', 'w')
file_data.create_dataset('X', data=X)
file_data.create_dataset('Y', data=Y)
file_data.create_dataset('Y_one_hot', data=Y_one_hot)
file_data.create_dataset('ids', data=ids)
file_data.create_dataset('genres', data=genres)
file_data.create_dataset('genres_unique', data=genres_unique)
file_data.close()
