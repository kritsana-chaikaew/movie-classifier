import numpy as np

from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt

import h5py
import sys
import glob
import os

def img2arr(img):
    arr = np.expand_dims(img_to_array(img), axis=0).astype('uint8')
    arr.resize(1, image_height, image_width, 3)
    return arr.copy()

dataset_name = str(input('Dataset name: '));

image_height = int(input('Image height: '))
image_width = image_height * 2 // 3
print('Image width: ', image_width)

X = np.empty((0, image_height, image_width, 3), dtype='uint8')
Y = np.empty((0,), dtype='uint8')
ids_genres = np.empty((0, 2), dtype='U')
genres_all = np.empty((0), dtype='U')
ids = np.empty((0,), dtype='U')

pattern = '../genre_list/' + ('[0-9]' * 6 + '-' + '[0-9]' * 6 + '.txt')
files = glob.glob(pattern)

if len(files) == 0:
    print('Download data first')
    exit()

for filename in files:
    array = np.genfromtxt(filename, delimiter=',', dtype='U')
    ids_genres = np.append(ids_genres, array, axis=0)
ids_genres = np.unique(ids_genres, axis=0)

ids_all = ids_genres[:, 0]
genre_sets = ids_genres[:, 1]

# one genre
for genre_set in genre_sets:
    genre = genre_set.split(':')[0]
    genres_all = np.append(genres_all, [genre], axis=0)

classes = np.unique(genres_all)

data_len = len(ids_all)
is_preview = False

num_data = int(input('Number of data: ') or data_len)
for i, id_, genre in zip(range(data_len), ids_all, genres_all):
    image_path = '../posters/' + id_ + '.jpg'

    if i >= num_data:
        break
    if not os.path.isfile(image_path):
        continue
    with open(image_path, 'rb') as f:
        image = Image.open(f)
        resized_image = image.resize(
                (image_width, image_height),
                Image.NEAREST)
        X = np.append(X, img2arr(resized_image), axis=0)
        Y = np.append(Y, np.where(classes==genre)[0].astype('uint8'), axis=0)
        ids = np.append(ids, [id_], axis=0)
        image.close()
        del resized_image
        del image

        if not is_preview:
            plt.imshow(X[0]/255)
            plt.title('ID '+id_+' Class '+str(Y[0]))
            plt.show()
            is_preview = True

    sys.stdout.write("\b"*20)
    sys.stdout.flush()
    sys.stdout.write("{0:.2f} %".format(i*100/data_len))
    sys.stdout.flush()

print(ids.shape)

if not os.path.exists('../datasets'):
    os.makedirs('../datasets')

if not os.path.exists('../datasets/'+dataset_name):
    os.makedirs('../datasets/'+dataset_name)

with h5py.File('../datasets/'+dataset_name+'/'+dataset_name+'.h5py', 'w') as file_data:
    file_data.create_dataset('X', data=X)
    file_data.create_dataset('Y', data=Y)

with open('../datasets/'+dataset_name+'/ids.txt', 'w') as file_ids:
    np.savetxt(file_ids, ids, fmt='%s', newline='\n')

with open('../datasets/classes.txt', 'w') as file_classes:
    np.savetxt(file_classes, classes, fmt='%s', newline='\n')
