import random


def normalize_data(steering_angle, center_images, right_images, left_images):

    probability_drop = 0.3
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

    return normalized_img, normalized_angles

