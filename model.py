import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
from keras.models import model_from_json

from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import pandas
import cv2

BATCH_SIZE = 64

log_file = 'data/driving_log.csv'

#Create list from CSV
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
input_data = pandas.read_csv(log_file, names=column_names)
center_images = input_data.center.tolist()[1:]
right_images = input_data.right.tolist()[1:]
left_images = input_data.left.tolist()[1:]
steering_angle = input_data.steering.tolist()[1:]


def normalise(X_train):
    a = -0.5
    b = 0.5
    X_normalized = a + (((X_train - np.min(X_train))*(b - a))/(np.max(X_train) - np.min(X_train)))
    return X_normalized


def resize_image(image):
    resized = cv2.resize(image, (132, 66), interpolation=cv2.INTER_LINEAR)
    return resized


def train_image_generator():

    batch_features = np.zeros((BATCH_SIZE, 66, 132, 3))
    batch_labels = np.zeros((BATCH_SIZE,),)
    while True:
        X_train,y_train = shuffle(center_images, steering_angle)
        for i in range(BATCH_SIZE):
            batch_features[i] = resize_image(mpimg.imread(path + X_train[i]))
            batch_labels[i] = y_train[i]
        yield batch_features, batch_labels


def validation_image_generator():
    batch_features = np.zeros((BATCH_SIZE, 66, 132, 3))
    batch_labels = np.zeros((BATCH_SIZE,),)
    while True:
        X_train,y_train = shuffle(center_images, steering_angle)
        for i in range(BATCH_SIZE):
            batch_features[i] = resize_image(mpimg.imread(path + X_train[i]))
            batch_labels[i] = y_train[i]
        yield batch_features, batch_labels



path = "data/"

shape = (66, 132, 3)

# Create the Sequential model
model = Sequential()

model.add(BatchNormalization(input_shape=shape))

model.add(Convolution2D(24, 5, 5, border_mode='valid'))

model.add(Convolution2D(36, 5, 5, border_mode='valid'))

model.add(Convolution2D(48, 5, 5, border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))

model.add(Flatten())

# 2nd Layer - Add a fully connected layer
model.add(Dense(100))

# 3rd Layer - Add a ReLU activation layer
model.add(Activation('tanh'))

# 4th Layer - Add a fully connected layer
model.add(Dense(50))

model.add(Activation('tanh'))

model.add(Dense(10))

model.add(Activation('tanh'))

model.add(Dense(1))

model.compile('adam', 'mean_squared_error')

model.summary()

valid_generator = validation_image_generator()
train_generator = train_image_generator()

nb_valid_samples = np.ceil((len(center_images)/5) * 0.2)
nb_samples_per_epoch = np.ceil((len(center_images))/5) - nb_valid_samples


print(nb_samples_per_epoch)
print(nb_valid_samples)


model.fit_generator(
        train_generator,
        samples_per_epoch=nb_samples_per_epoch,
        nb_epoch=5,
        validation_data=valid_generator,
        nb_val_samples=nb_valid_samples)


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Saved model to disk")





