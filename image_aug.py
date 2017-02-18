import cv2

def crop_image(image):
    cropped = image[59:image.shape[0] - 26, :]
    return cropped


def resize_image(image):
    resized = cv2.resize(image, (200, 66), interpolation=cv2.INTER_LINEAR)
    return resized


def flip_image(image):
    flipped_img = cv2.flip(image, 1)
    return flipped_img


def convert_to_yuv(image):
    yuv_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    return yuv_image