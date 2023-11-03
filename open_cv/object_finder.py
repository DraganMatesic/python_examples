import os
import cv2
import numpy as np

# REQUIREMENTS
# opencv-python
# numpy


def object_counter(template_dir, target_image, threshold=0.90, distance: int = None, get_objects=False):
    """
    :param template_dir: directory where template images are located
    :param target_image: path to image with whom we want to compare 1 or more template images from template dir
    :param threshold: what is allowed threshold for difference of template image while searching similarities on target image
    :param distance: if integer is provided it will exclude all template matches where new cords are close X px to existing cords
    :param get_objects: when set True it will return coordinates (x,y) where template match was found else it will return number of objects it found
    :return:
    """
    discovered_objects = list()
    img_rgb = cv2.imread(target_image)

    images = os.listdir(template_dir)
    for image in images:
        image_path = os.path.join(template_dir, image)
        template = cv2.imread(image_path)

        res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if loc[0].size > 0:
            for x, y in list(zip(loc[1], loc[0])):
                pack_cords = (x, y)

                if distance is not None:
                    # calculate if cords of current image is closer than X px from other cords in list
                    exists = [(x1, y1) for x1, y1 in discovered_objects
                              if abs(x1 - x) <= distance or abs(y1 - y) <= distance]
                    if exists:
                        continue

                discovered_objects.append(pack_cords)

    if get_objects is True:
        return discovered_objects

    return len(discovered_objects)
