import numpy as np

from keras.preprocessing.image import img_to_array
from PIL import Image
# import matplotlib.pyplot as plt

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
is_gen_classes = str(input('Generate class file? [y/n]: ') or 'y')
num_data = int(input('Number of data [all]: ') or -1)

X = np.empty((0, image_height, image_width, 3), dtype='uint8')
Y = np.empty((0,), dtype='uint8')
data_list = np.empty((0, 2), dtype='U')
ids = np.empty((0,), dtype='U')

files = glob.glob('../genre_list/all.txt')

if len(files) == 0:
    print('Download data first')
    exit()

for filename in files:
    array = np.genfromtxt(filename, delimiter=',', dtype='U')
    data_list = np.append(data_list, array, axis=0)
data_list = np.unique(data_list, axis=0)

id_list = data_list[:, 0]
genre_list = data_list[:, 1]

data_len = len(id_list)
num_data = num_data == -1 and data_len or num_data

interest_genres = [
        # 'Action',
        # 'Adventure',
        'Animation',
        # 'Biography',
        # 'Comedy',
        # 'Crime',
        # 'Drama',
        # 'Family',
        # 'Fantasy',
        'Horror']
        # 'Mystery',
        # 'Sci-Fi',
        # 'Thriller']
classes = np.unique(interest_genres)

# is_preview = False

zipped = zip(range(data_len), id_list, genre_list)
for i, id_, genre_set in zipped:

    if i >= num_data:
        break

    for genre in genre_set.split(':'):
        if genre in interest_genres:
            try:
                image_path = '../posters/' + id_ + '.jpg'
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

                    # if not is_preview:
                    #     plt.imshow(X[0]/255)
                    #     plt.title('ID '+id_+' Class '+str(Y[0]))
                    #     plt.show()
                    #     is_preview = True

                sys.stdout.write("\b"*20)
                sys.stdout.flush()
                sys.stdout.write("{0:.2f} %".format(i*100/data_len))
                sys.stdout.flush()

            except:
                pass

            break

print('\n'+str(ids.shape))

if not os.path.exists('../datasets'):
    os.makedirs('../datasets')

if not os.path.exists('../datasets/'+dataset_name):
    os.makedirs('../datasets/'+dataset_name)

with h5py.File('../datasets/'+dataset_name+'/'+dataset_name+'.h5py', 'w') as file_data:
    file_data.create_dataset('X', data=X)
    file_data.create_dataset('Y', data=Y)

with open('../datasets/'+dataset_name+'/ids.txt', 'w') as file_ids:
    np.savetxt(file_ids, ids, fmt='%s', newline='\n')

if is_gen_classes == 'y':
    with open('../datasets/classes.txt', 'w') as file_classes:
        np.savetxt(file_classes, classes, fmt='%s', newline='\n')
