import json
import argparse
import base64
import image_aug
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from datetime import datetime
import os
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
from keras.models import model_from_json

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
step_size_param = 0.2
prev_angle = 0
image_folder = 'run/images'

@sio.on('telemetry')
def telemetry(sid, data):
    try:
        global prev_angle
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)

        # Apply same augmentations as in traingin set
        cropped_image = image_aug.crop_image(image_array)
        resized_image = image_aug.resize_image(cropped_image)
        yuv_image = image_aug.convert_to_yuv(resized_image)
        transformed_image_array = yuv_image[None, :, :, :]
        steering_angle = float(model.predict(transformed_image_array, batch_size=1))

        # Applies smoothing 1st degree filter to angle
        next_angle = steering_angle * step_size_param + (1-step_size_param) * prev_angle

        # Get value for throttle linerally proportional to angle
        throttle = abs(next_angle) * -0.01 + 0.15
        prev_angle = next_angle
        print(next_angle, throttle)
        # Send control
        send_control(next_angle, throttle)

        # Save image for later use
        timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        image_filename = os.path.join(image_folder, timestamp)
        image.save('{}.jpg'.format(image_filename))

    except Exception as e:
        print(e)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # Load model
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    print("Creating image folder at {}".format(image_folder))

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
