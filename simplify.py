import cv2
import glob
import os
import shutil
import math
import numpy as np
from pathlib import Path
import separation

im_dir = 'images/'
new_dir = 'modified_images/'
image_width = 32
image_height = 32


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def show_image(image):
    cv2.imshow('show', image)
    cv2.waitKey(0)


def crop(image):
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (xmin, ymin, hmax, wmax) = (math.inf, math.inf, 0, 0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        (xmin, ymin) = (min(x, xmin), min(y, ymin))
        (hmax, wmax) = (max(hmax, h + y), max(wmax, w + x))

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    ROI = thresh[ymin:hmax, xmin:wmax]

    # _, thresh = cv2.threshold(ROI, 127, 255, cv2.THRESH_BINARY_INV)
    # kernel = np.ones((3, 3), 'uint8')
    # erode = cv2.erode(thresh, kernel, iterations=1)
    # closing = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    # _, thresh = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY_INV)
    return ROI


def remove_alpha_channel(image):
    if np.ndim(image) == 3:
        if np.size(image, 2) == 4:
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
    return image


def simplify_image(image) -> str:
    image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    remove_alpha_channel(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_img = crop(gray)
    return crop_img


def rotate(image):
    (h, w) = image.shape[:2]
    image_center = (w // 2, h // 2)
    angels = (5, -5, 10, -10)
    rotate_image = []
    for angle in angels:
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        abs_cos = np.abs(rotation_mat[0, 0])
        abs_sin = np.abs(rotation_mat[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        rotated = cv2.warpAffine(image, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
        cropped = crop(rotated)
        rotate_image.append(cropped)
    return rotate_image


def balancing():
    arr_len_files = []
    for folder in glob.glob(f'{new_dir}*'):
        path = Path(f'{folder}')
        arr_len_files.append(sum(1 for x in path.glob('*') if x.is_file()))

    min_value = min(arr_len_files)
    for folder in glob.glob(f'{new_dir}*'):
        arr = []
        for image in os.listdir(folder):
            arr.append(folder + '/' + image)
        d = 0
        k = len(arr)
        for i in arr:
            if d == k - min_value:
                break
            os.remove(i)
            d += 1
    return min_value


def start() -> None:
    create_directory(new_dir)

    i = 0
    for folder in glob.glob(f'{im_dir}*'):
        print(folder)
        letter = '{0:06b}'.format(i)
        Path(f"{new_dir}{letter}").mkdir(parents=True, exist_ok=True)
        for image in glob.glob(f'{folder}/*.png'):
            file_without_extension = image.split('.')[0]

            new_image = [simplify_image(image)]
            rotate_image = rotate(new_image[0])
            new_image = new_image + rotate_image
            k = 0
            for n_image in new_image:
                path = new_dir + letter + '/' + file_without_extension.split('\\')[2] + f'{k}'
                n_image = cv2.resize(n_image, (image_width, image_height))
                cv2.imwrite(path + '.jpeg', n_image)
                k += 1
        i += 1
    min_val = balancing()

    separation.start(min_val)


if __name__ == '__main__':
    start()
