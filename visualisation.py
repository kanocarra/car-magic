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
throttle = input_data.throttle.tolist()


def show_data():
    angles = np.array(steering_angle)
    rounded_angles = list(np.round(angles, decimals=2))
    rounded_angles.count(0.75)
    labels = set(rounded_angles)
    totals = []
    for angle in labels:
        totals.append(rounded_angles.count(angle))

    totals.sort()
    # the histogram of the data
    plt.hist(rounded_angles, len(labels), normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('Steering Angle')
    plt.ylabel('Probability')
    plt.title('Histogram of Steering angle occurences:')
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


def normalize_data():

    global steering_angle, center_images
    probability_drop = 0.2
    probability_right_camera = 0.5
    normalized_angles = []
    normalized_img = []
    steering_angle.count(0.75)
    for angle, img_center, img_right, img_left in zip(steering_angle, center_images, right_images, left_images):
        prob_value = random.random()
        if abs(angle) < 0.01:
            if prob_value < probability_drop:
                normalized_angles.append(angle)
                normalized_img.append(img_center)
        elif not abs(angle) > 0.6:
            prob_value = random.random()
            normalized_angles.append(angle)
            normalized_img.append(img_center)

            prob_value = random.random()

            if probability_right_camera >= prob_value:
                prob_value = random.random()
                if probability_right_camera >= prob_value :
                    new_angle = angle - 0.25
                    normalized_angles.append(new_angle)
                    normalized_img.append(img_right)
                else:
                    new_angle = angle + 0.25
                    normalized_angles.append(new_angle)
                    normalized_img.append(img_left)

    steering_angle = normalized_angles

    show_data()



#def steering_vs_throttle():


#show_data()
normalize_data()