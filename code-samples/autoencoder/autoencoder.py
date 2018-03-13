import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Flatten, Input, Reshape
from keras.optimizers import Adam

from sklearn import datasets

import math
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

RANDO = 42
SMILING = [0, 7, 8, 3, 11, 12, 13, 14, 20, 27, 155, 153, 154, 297]
NONSMILING = [1, 2, 3, 6, 10, 60, 61, 136, 138, 216, 219, 280]


# faces = datasets.fetch_olivetti_faces()
faces = datasets.fetch_lfw_people(data_home='.')

# # smiling/nonsmiling
# fig = plt.figure(figsize=(5, 3))
# for i in range(len(SMILING)):
#     ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(faces.images[SMILING[i], :, :], cmap='gray')
#
# plt.savefig('smiling-faces.pdf')
#
# fig = plt.figure(figsize=(5, 3))
# for i in range(len(NONSMILING)):
#     ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
#     ax.imshow(faces.images[NONSMILING[i], :, :], cmap='gray')
#
# plt.savefig('nonsmiling-faces.pdf')

from keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, MaxPooling2D, UpSampling2D, Flatten, Cropping2D
from keras.models import Model, Sequential

hidden_size = 64

encoder = Sequential()

encoder.add(Flatten(input_shape=(62, 47, 1)))
encoder.add(Dense(1024, activation='relu'))
encoder.add(Dense(512, activation='relu'))
encoder.add(Dense(256, activation='relu'))
encoder.add(Dense(128, activation='relu'))
encoder.add(Dense(hidden_size))

decoder = Sequential()

decoder.add(Dense(128, activation='relu', input_dim=64))
decoder.add(Dense(256, activation='relu'))
decoder.add(Dense(512, activation='relu'))
decoder.add(Dense(1024, activation='relu'))
decoder.add(Dense(62*47, activation='relu'))
decoder.add(Reshape((62,47,1)))

auto = Sequential()

auto.add(encoder)
auto.add(decoder)

auto.summary()

optimizer = Adam(lr=0.001)
auto.compile(optimizer=optimizer, loss='mse')

x = faces.images[:, :, :, None] / 255.

for e in range(1):
    auto.fit(x, x, epochs=5, batch_size=256, shuffle=True)

    out = auto.predict(x[:400, :])

    fig = plt.figure(figsize=(5, 6))

    # plot several images
    for i in range(30):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(out[i, ...].reshape((62, 47)), cmap=plt.cm.gray)

    plt.savefig('faces-reconstructed.{:03d}.pdf'.format(e))

smiling = x[SMILING, ...]
nonsmiling = x[NONSMILING, ...]

smiling_latent = encoder.predict(smiling)
nonsmiling_latent = encoder.predict(nonsmiling)

smiling_mean = smiling_latent.mean(axis=0)
nonsmiling_mean = nonsmiling_latent.mean(axis=0)

smiling_vector = smiling_mean - nonsmiling_mean

rando_latent = encoder.predict(x[None, RANDO, ...])

fig = plt.figure(figsize=(5, 1))

# plot several images
for add in np.linspace(0.0, 1.0, 30):

    gen_latent = rando_latent + add * smiling_vector
    gen = decoder.predict(gen_latent)

    ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(gen.reshape((62, 47)), cmap=plt.cm.gray)

plt.savefig('rando-to-smiling.pdf'.format(e))



