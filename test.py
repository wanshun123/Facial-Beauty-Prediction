# https://towardsdatascience.com/how-attractive-are-you-in-the-eyes-of-deep-neural-network-3d71c0755ccc

from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# import cv2
# im = cv2.imread("abc.tiff")

import cv2
import glob
import numpy as np

images = []
files = glob.glob ("C:/Users/Miles/Desktop/python/faces/SCUT-FBP5500_v2/Images/*.jpg")
for myFile in files:
    print(myFile)
    image = cv2.imread(myFile)
    images.append(image)

print('images shape:', np.array(images).shape)

from io import StringIO
labels = np.genfromtxt('All_labels_alphabetized_nolabel.txt')

#np.random.seed(0)
#msk = np.random.rand(len(images)) < 0.8

#indices = np.random.permutation(5500)
#training_idx, test_idx = indices[:4400], indices[4400:]
#labels_train, labels_test = labels[training_idx], labels[test_idx]

#train = df[msk]
#test = df[~msk]

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)

resnet = ResNet50(include_top=False, pooling='avg')
model = Sequential()
model.add(resnet)
model.add(Dense(1))
model.layers[0].trainable = False
model.compile(loss='mean_squared_error', optimizer=Adam())

images_train = np.array(images_train)
labels_train = np.array(labels_train)
images_test = np.array(images_test)
labels_test = np.array(labels_test)

model.fit(batch_size = 32, x = images_train, y = labels_train, epochs = 6)

from keras.models import load_model
model = load_model('my_model.h5')

results = model.evaluate(images_test, labels_test)
predict = model.predict(images_test)

#np.mean(predict - labels_test)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
mae = mean_absolute_error(labels_test, predict)
rmse = mean_squared_error(labels_test, predict)

