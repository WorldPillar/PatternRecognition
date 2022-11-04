import cv2
import glob
import os
import shutil
import math
import numpy as np
from pathlib import Path

im_dir = 'images/'
new_dir = 'modified_images/'
image_width = 32
image_height = 32


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    (xmin, ymin, hmax, wmax) = (math.inf, math.inf, 0, 0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        (xmin, ymin) = (min(x, xmin), min(y, ymin))
        (hmax, wmax) = (max(hmax, h + y), max(wmax, w + x))

    ROI = image[ymin:hmax, xmin:wmax]
    ROI = cv2.resize(ROI, (image_width, image_height))
    return ROI


def simplify_image(image, letter) -> str:
    file_without_extension = image.split('.')[0]
    image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    trans_mask = image[:, :, 3] == 0
    image[trans_mask] = [255, 255, 255, 255]
    crop_img = crop(image)
    path = new_dir + letter + '/' + file_without_extension.split('\\')[2]
    cv2.imwrite(path + '.jpeg', crop_img)
    return path + '.jpeg'


def rotate(image):
    file_without_extension = image.split('.')[0]
    img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    (h, w) = img.shape[:2]
    image_center = (w // 2, h // 2)
    angels = (10, -10)
    for angle in angels:
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

        abs_cos = np.abs(rotation_mat[0, 0])
        abs_sin = np.abs(rotation_mat[0, 1])

        bound_w = int(h * abs_sin + w * abs_cos)
        bound_h = int(h * abs_cos + w * abs_sin)

        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]

        rotated = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), borderValue=(255, 255, 255))
        cropped = crop(rotated)
        cv2.imwrite(file_without_extension +
                    str(angle) + '.jpeg', cropped)


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


def start() -> None:
    create_directory(new_dir)

    i = 0
    for folder in glob.glob(f'{im_dir}*'):
        letter = '{0:06b}'.format(i)
        Path(f"{new_dir}{letter}").mkdir(parents=True, exist_ok=True)
        for image in glob.glob(f'{folder}/*.png'):
            new_path = simplify_image(image, letter)
            rotate(new_path)
        i += 1
    balancing()


if __name__ == '__main__':
    start()
