import numpy as np
import matplotlib.image as mpimg
import image_aug
import data_aug
import random
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas
import random

BATCH_SIZE = 32

log_file = 'data/driving_log.csv'

#Create list from CSV
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
input_data = pandas.read_csv(log_file, names=column_names)
center_images = input_data.center.tolist()[1:]
right_images = input_data.right.tolist()[1:]
left_images = input_data.left.tolist()[1:]
steering_angle = input_data.steering.tolist()[1:]
shape = (47, 200, 3)


X_normalized, y_normalized = data_aug.normalize_data(steering_angle, center_images, right_images, left_images)

X_train, X_validation, y_train, y_validation = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=0)


def edit_path(path):
    index = path.find('data')
    new_path = path[index:]
    return new_path


def train_image_generator():

    probability = 0.5
    global X_train, y_train
    batch_image = np.zeros((BATCH_SIZE, 47, 200, 3))
    batch_angle = np.zeros((BATCH_SIZE,),)
    while True:
        X_train,y_train = shuffle(X_train, y_train)
        flip_probability = 0.5
        for i in range(BATCH_SIZE):
            index = random.randint(0, len(X_train)-1)
            path = edit_path(X_train[index])
            cropped_image = image_aug.crop_image(mpimg.imread(path))
            resized_image = image_aug.resize_image(cropped_image)
            prob_value = random.random()
            if prob_value >= flip_probability:
                flipped_image = image_aug.flip_image(resized_image)
                batch_image[i] = flipped_image
                batch_angle[i] = y_train[index] * -1
            else:
                batch_image[i] = resized_image
                batch_angle[i] = y_train[index]

        yield batch_image, batch_angle


def validation_image_generator():

    global X_validation, y_validation
    batch_image = np.zeros((BATCH_SIZE, 47, 200, 3))
    batch_angle = np.zeros((BATCH_SIZE,),)

    while True:
        X_validation,y_validation = shuffle(X_validation, y_validation)
        flip_probability = 0.5
        for i in range(BATCH_SIZE):
            index = random.randint(0, len(X_validation)-1)
            path = edit_path(X_validation[index])
            cropped_image = image_aug.crop_image(mpimg.imread(path))
            resized_image = image_aug.resize_image(cropped_image)
            prob_value = random.random()
            if prob_value >= flip_probability:
                flipped_image = image_aug.flip_image(resized_image)
                batch_image[i] = flipped_image
                batch_angle[i] = y_validation[index] * -1
            else:
                batch_image[i] = resized_image
                batch_angle[i] = y_validation[index]
        yield batch_image, batch_angle



# Create the Sequential model
model = Sequential()

model.add(BatchNormalization(input_shape=shape))

model.add(Convolution2D(24, 5, 5, border_mode='valid'))

model.add(Convolution2D(36, 5, 5, border_mode='valid'))

model.add(Convolution2D(48, 5, 5, border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))

model.add(Dropout(0.5))

model.add(Activation('relu'))

model.add(Flatten())

model.add(Activation('relu'))

model.add(Dense(100))

model.add(Activation('relu'))

model.add(Dense(50))

model.add(Activation('relu'))

model.add(Dense(10))

model.add(Dropout(0.5))

model.add(Activation('tanh'))

model.add(Dense(1))

model.compile('adam', 'mean_squared_error')

model.summary()

valid_generator = validation_image_generator()
train_generator = train_image_generator()

nb_samples_per_epoch = np.ceil(len(X_train) * 1.4 /BATCH_SIZE) * BATCH_SIZE
nb_valid_samples = np.ceil(nb_samples_per_epoch * 0.2)


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