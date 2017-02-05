import random
import pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

log_file = 'data/driving_log.csv'
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
input_data = pandas.read_csv(log_file, names=column_names)
center_images = input_data.center.tolist()
steering_angle = input_data.steering.tolist()[1:]
right_images = input_data.right.tolist()
left_images = input_data.left.tolist()


def show_data():
    angles = np.array(steering_angle)
    rounded_angles = list(np.round(steering_angle, decimals=2))
    labels = set(rounded_angles)
    totals = []
    for angle in labels:
        totals.append(rounded_angles.count(angle))

    totals.sort()
    # the histogram of the data
    plt.hist(rounded_angles, len(labels), normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('Steering Angle')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Steering angle occurences:}\$')
    plt.grid(True)
    plt.show()
    print("Plotted")



def crop_image():
    for i in range(0, 20):
        num = random.randint(0, len(right_images))
        image = mpimg.imread(right_images[num])
        cropped = image[59:image.shape[0] - 26, :]
        print(cropped.shape)
        plt.figure()
        plt.imshow(cropped)
        plt.title(steering_angle[num])
        print("Done image.")


show_data()