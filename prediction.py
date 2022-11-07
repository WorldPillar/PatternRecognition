import numpy as np
import glob
import os
import shutil
import cv2
import math
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('model.h5')
im_dir = 'input_dataset/'
res_dir = 'output_results/'
image_width = 32
image_height = 32


def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def show_image(image):
    cv2.imshow('show', image)
    cv2.waitKey(0)


def find_contours(contours, one_letter):
    bounding = []
    (x_min, y_min, h_max, w_max) = (math.inf, math.inf, 0, 0)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = h / float(w)
        if 0.6 <= ar <= 5:
            if one_letter:
                (x_min, y_min, h_max, w_max) = (min(x, x_min), min(y, y_min),
                                                max(h_max, h + y), max(w_max, w + x))
            else:
                bounding.append((x, y, h + y, w + x))

    if one_letter:
        bounding.append((x_min, y_min, h_max, w_max))
    return bounding


def gamma_correct(image):
    new_image = cv2.convertScaleAbs(image, alpha=1.9, beta=10)
    return new_image


def crop(image, one_letter):
    # image = gamma_correct(image)
    blurred = cv2.GaussianBlur(image, (5, 5), 1)
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    # show_image(dilation)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    bounding = find_contours(contours, one_letter)
    return bounding


def get_image(path):
    image = cv2.imread(path, cv2.COLOR_RGBA2RGB)
    if np.ndim(image) == 3:
        if np.size(image, 2) == 4:
            trans_mask = image[:, :, 3] == 0
            image[trans_mask] = [255, 255, 255, 255]
    return image


def get_letter(image, bound):
    letter = image[bound[1]:bound[2], bound[0]:bound[3]]

    blurred = cv2.GaussianBlur(letter, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY_INV)

    letter = cv2.resize(thresh, (image_width, image_height))
    return letter


def print_results(classes):
    result = ""
    for i in range(33):
        if classes[i] > 0.1:
            result += f"{print_letter(i)} - {classes[i] * 100}%\n"
    print(result)


def predicting(path_to_image):
    image = get_image(path_to_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounding = crop(gray, False)

    im2 = image.copy()
    for bound in bounding:
        letter = get_letter(gray, bound).astype(np.uint8) / 255.
        letter = np.expand_dims(letter, axis=0)
        letter = np.vstack([letter])
        classes = model(letter)
        print_results(classes[0])

        (x, y, h, w) = (bound[0], bound[1], bound[2], bound[3])
        i = int(np.argmax(classes))
        cv2.rectangle(im2, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(im2, f'{i}', (x + 10, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    im_path = path_to_image.split('.')[0].split('\\')[1]
    cv2.imwrite(res_dir + im_path + '_det' + '.jpeg', im2)


def files_predicting():
    np.set_printoptions(precision=3)
    for image in glob.glob(f'{im_dir}*'):
        print(image)
        predicting(image)


def start() -> None:
    create_directory(res_dir)
    files_predicting()


if __name__ == '__main__':
    start()
