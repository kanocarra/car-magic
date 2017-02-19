import numpy as np
import matplotlib.image as mpimg
import image_aug
import data_aug
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas
import random

BATCH_SIZE = 32

log_file = 'fine_tune_data/driving_log.csv'

# Read in driving_log CSV data
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
input_data = pandas.read_csv(log_file, names=column_names)
center_images = input_data.center.tolist()[1:]
right_images = input_data.right.tolist()[1:]
left_images = input_data.left.tolist()[1:]
steering_angle = input_data.steering.tolist()[1:]

# Define the shape of the input data
shape = (66, 200, 3)

# Normalize the data
X_normalized, y_normalized = data_aug.normalize_data(steering_angle, center_images, right_images, left_images)

# Seperate into training and validation sets
X_train, X_validation, y_train, y_validation = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=0)


# Removes full path from file name
def edit_path(path):
    index = path.find('fine_tune_data')
    new_path = path[index:]
    return new_path


# Generator for the training data
def train_image_generator():

    global X_train, y_train

    # Probability that an image will be flipped
    flip_probability = 0.5

    # Initialise arrays for batch size
    batch_image = np.zeros((BATCH_SIZE, 66, 200, 3))
    batch_angle = np.zeros((BATCH_SIZE,),)

    # Loops trhough data set until batch size fulfilled
    while True:
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(BATCH_SIZE):

            # Pick a random index
            index = random.randint(0, len(X_train)-1)

            # Adjusts the path for running on AWS instance
            if X_train[index].find('kanocarra') > -1 :
                path = edit_path(X_train[index]).replace(' ', '')
            else:
                path = ('fine_tune_data/' + X_train[index]).replace(' ', '')

            # Crop out unimportant features
            cropped_image = image_aug.crop_image(mpimg.imread(path))

            # Resize image to fit NVIDIA model
            resized_image = image_aug.resize_image(cropped_image)

            # Convert to YUV
            yuv_image = image_aug.convert_to_yuv(resized_image)

            prob_value = random.random()

            # Determine whether to flip the image or not and save into batch
            if prob_value >= flip_probability:
                flipped_image = image_aug.flip_image(yuv_image)
                batch_image[i] = flipped_image
                batch_angle[i] = y_train[index] * -1
            else:
                batch_image[i] = yuv_image
                batch_angle[i] = y_train[index]

        yield batch_image, batch_angle


# Generator for the validation data
def validation_image_generator():

    global X_validation, y_validation

    # Probability that an image will be flipped
    flip_probability = 0.5

    # Initialise arrays for batch size
    batch_image = np.zeros((BATCH_SIZE, 66, 200, 3))
    batch_angle = np.zeros((BATCH_SIZE,),)

    while True:
        X_validation,y_validation = shuffle(X_validation, y_validation)
        for i in range(BATCH_SIZE):

            # Pick a random index
            index = random.randint(0, len(X_validation)-1)

            # Adjusts the path for running on AWS instance
            if X_validation[index].find('kanocarra') > -1 :
                path = edit_path(X_validation[index]).replace(' ', '')
            else:
                path = ('fine_tune_data/' + X_validation[index]).replace(' ', '')

            # Crop out unimportant features
            cropped_image = image_aug.crop_image(mpimg.imread(path))

            # Resize image to fit NVIDIA model
            resized_image = image_aug.resize_image(cropped_image)

            # Convert to YUV
            yuv_image = image_aug.convert_to_yuv(resized_image)

            prob_value = random.random()

            # Determine whether to flip the image or not and save into batch
            if prob_value >= flip_probability:
                flipped_image = image_aug.flip_image(yuv_image)
                batch_image[i] = flipped_image
                batch_angle[i] = y_validation[index] * -1
            else:
                batch_image[i] = yuv_image
                batch_angle[i] = y_validation[index]
        yield batch_image, batch_angle


# Generator for the test data
def test_image_generator():

    global X_test, y_test

    # Initialise arrays for batch size
    batch_image = np.zeros((BATCH_SIZE, 66, 200, 3))
    batch_angle = np.zeros((BATCH_SIZE,),)

    # Put random images into test batches
    while True:
        X_test,y_test = shuffle(X_test, y_test)
        for i in range(BATCH_SIZE):
            index = random.randint(0, len(X_test)-1)
            path = ('fine_tune_data/' + X_test[index]).replace(' ', '')
            cropped_image = image_aug.crop_image(mpimg.imread(path))
            resized_image = image_aug.resize_image(cropped_image)
            batch_image[i] = resized_image
            batch_angle[i] = y_test[index]
        yield batch_image, batch_angle


# Create the Sequential model
model = Sequential()

# Add normalization layer
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=shape, name='Normalization'))

# Add 2D convolution with 5x5 kernel and 2x2 stride
model.add(Convolution2D(24, 5, 5, border_mode='valid', name="conv1", activation='elu', init='he_normal', subsample=(2,2)))

# Apply dropout
model.add(SpatialDropout2D(0.2))

# Add 2D convolution with 5x5 kernel and 2x2 stride
model.add(Convolution2D(36, 5, 5, border_mode='valid', name="conv2", activation='elu', subsample=(2,2)))

# Apply dropout
model.add(SpatialDropout2D(0.2))

# Add 2D convolution with 5x5 kernel and 2x2 stride
model.add(Convolution2D(48, 5, 5, border_mode='valid', name="conv3", activation='elu', subsample=(2,2)))

# Apply dropout
model.add(SpatialDropout2D(0.2))

# Add 2D convolution with 3x3 kernel and 1x1 stride
model.add(Convolution2D(64, 3, 3, border_mode='valid', name="conv4", activation='elu', subsample=(1,1)))

model.add(SpatialDropout2D(0.2))

# Add 2D convolution with 3x3 kernel and 1x1 stride
model.add(Convolution2D(64, 3, 3, border_mode='valid', name="conv5", activation='elu', subsample=(1,1)))

# Apply dropout
model.add(SpatialDropout2D(0.2))

# Flatten layers
model.add(Flatten(name="flatten"))

# Fully connected layer with elu activation
model.add(Dense(100, name="dense1", activation='elu'))

# Apply dropout
model.add(Dropout(0.5))

# Fully connected layer with elu activation
model.add(Dense(50, name="dense2", activation='elu'))

# Apply dropout
model.add(Dropout(0.5))

# Fully connected layer with elu activation
model.add(Dense(10, name="dense3", activation='elu'))

# Apply dropout
model.add(Dropout(0.5))

# Fully connected layer with elu activation
model.add(Dense(1))

# Add Adam optimiser with mse optimisation
model.compile('adam', 'mean_squared_error')

# Load previous weights
model.load_weights('model.h5')

# Print out model summary
model.summary()

# Define the generators
valid_generator = validation_image_generator()
train_generator = train_image_generator()

# Define number of samples for epochs
nb_samples_per_epoch = np.ceil(len(X_train) * 2 / BATCH_SIZE) * BATCH_SIZE
nb_valid_samples = np.ceil(nb_samples_per_epoch * 0.2)

print('Number of training smaples: ', nb_samples_per_epoch)
print('Number of validation samples: ', nb_valid_samples)

# Apply the generators
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_samples_per_epoch,
        nb_epoch=5,
        validation_data=valid_generator,
        nb_val_samples=nb_valid_samples)

# Save model to json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save model weights
model.save_weights("model.h5")
print("Saved mode.")

