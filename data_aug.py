# Collection of methods for data augmentation

import random


# Normalises the data by reducing number of low angles
def normalize_data(steering_angle, center_images, right_images, left_images):

    # Probabbility of dropping data
    probability_drop = 0.5

    # Probabbility of taking right camera image
    probability_right_camera = 0.5

    # Initialise arrays for normalised data
    normalized_angles = []
    normalized_img = []

    for angle, img_center, img_right, img_left in zip(steering_angle, center_images, right_images, left_images):
        prob_value = random.random()
        angle = float(angle)

        # Randomly drop angles less than 0.01 with likelihood of probability_drop
        if abs(angle) < 0.01:
            if prob_value < probability_drop:
                normalized_angles.append(angle)
                normalized_img.append(img_center)

                prob_value = random.random()

                # Add the right camera image with probability of probability_right_camera else add left image
                if probability_right_camera >= prob_value:
                    # Adjust angle by -0.25
                    new_angle = angle - 0.25
                    normalized_angles.append(new_angle)
                    normalized_img.append(img_right)
                else:
                    # Adjust angle by +0.25
                    new_angle = angle + 0.25
                    normalized_angles.append(new_angle)
                    normalized_img.append(img_left)
        elif not abs(angle) > 0.7:

            normalized_angles.append(angle)
            normalized_img.append(img_center)

            prob_value = random.random()

            # Add the right camera image with probability of probability_right_camera else add left image
            if probability_right_camera >= prob_value:
                # Adjust angle by -0.25
                new_angle = angle - 0.25
                normalized_angles.append(new_angle)
                normalized_img.append(img_right)
            else:
                # Adjust angle by +0.25
                new_angle = angle + 0.25
                normalized_angles.append(new_angle)
                normalized_img.append(img_left)

    return normalized_img, normalized_angles

