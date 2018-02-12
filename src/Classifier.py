import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

genres = np.genfromtxt('../genres.txt', delimiter=',', dtype=None)
poster_ids = np.genfromtxt('../poster_ids.txt', dtype=None)
genres = genres[:, 1]

genres_unique = np.unique(genres)
d = {}
for i, genre in enumerate(genres_unique):
    d[genre] = i

l = [d[g] for g in genres]
print(len(l))

nClasses = len(genres_unique)
genre_one_hot = to_categorical(l)
print(genre_one_hot)
