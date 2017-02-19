# Collection of methods for image augmentation

import cv2


# Crops out landscape and car bonnet
def crop_image(image):
    cropped = image[59:image.shape[0] - 26, :]
    return cropped


# Resizes image to fit NVIDIA model
def resize_image(image):
    resized = cv2.resize(image, (200, 66), interpolation=cv2.INTER_LINEAR)
    return resized


# Flips image around vertical axis
def flip_image(image):
    flipped_img = cv2.flip(image, 1)
    return flipped_img


# Converts from RGB to YUV
def convert_to_yuv(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    return yuv_image