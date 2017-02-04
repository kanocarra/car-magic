import random
import pandas
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

log_file = 'data/driving_log.csv'
column_names = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
input_data = pandas.read_csv(log_file, names=column_names)
center_images = input_data.center.tolist()
steering_angle = input_data.steering.tolist()
right_images = input_data.right.tolist()
left_images = input_data.left.tolist()

path = 'data/'


def crop_image():
    for i in range(0, 20):
        num = random.randint(0, len(right_images))
        image = mpimg.imread(path + right_images[num])
        cropped = image[59:image.shape[0] - 26, :]
        print(cropped.shape)
        plt.figure()
        plt.imshow(cropped)
        plt.title(steering_angle[num])
        print("Done image.")


crop_image()