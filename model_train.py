from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import cv2
import glob
import numpy as np
import os
from io import StringIO

images_path = os.path.expanduser('~/Images')

images = []
files = glob.glob(images_path + '/*.jpg')
for myFile in sorted(files):
    image = cv2.imread(myFile)
    images.append(image)

labels = np.genfromtxt('All_labels_alphabetized_nolabel.txt')

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)

images_train = np.array(images_train)
labels_train = np.array(labels_train)
images_test = np.array(images_test)
labels_test = np.array(labels_test)

resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(1))
model.layers[0].trainable = False
model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(batch_size = 32, x = images_train, y = labels_train, epochs = 30)

model.layers[0].trainable = True
model.compile(loss='mean_squared_error', optimizer=Adam())
model.fit(batch_size = 32, x = images_train, y = labels_train, epochs = 30)

model.save('model.h5')

predict = model.predict(images_test, verbose = 1)
