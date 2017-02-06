import cv2

def crop_image(image):
    cropped = image[59:image.shape[0] - 26, :]
    return cropped


def resize_image(image):
    resized = cv2.resize(image, (200, 47), interpolation=cv2.INTER_LINEAR)
    return resized


def flip_image(image):
    flipped_img = cv2.flip(image, 1)
    return flipped_img

