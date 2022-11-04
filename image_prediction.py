import numpy as np
import glob
import os
import cv2
import math
from tensorflow import keras

im_dir = 'my_images/'
image_width = 32
image_height = 32


def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]


def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    boundings = []
    (xmin, ymin, hmax, wmax) = (math.inf, math.inf, 0, 0)
    im2 = image.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = h / float(w)
        if 0.6 <= ar <= 6:
            (xmin, ymin, hmax, wmax) = (min(x, xmin), min(y, ymin),
                                        max(hmax, h + y), max(wmax, w + x))
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('final', im2)
    cv2.waitKey(0)
    ROI = gray[ymin:hmax, xmin:wmax]
    ROI = cv2.resize(ROI, (image_width, image_height))
    return ROI


def simplify_image(image):
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    crop_img = crop(image)
    new_img = cv2.cvtColor(crop_img, cv2.IMREAD_COLOR)
    return new_img


def predicting(path_to_image):
    model = keras.models.load_model('model.h5')

    img = simplify_image(path_to_image).astype(np.float32) / 255.
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    classes = model.predict(img, batch_size=1)
    return classes[0]


def files_predicting():
    np.set_printoptions(precision=3)
    result = []
    for image in glob.glob(f'{im_dir}*'):
        print(image)
        result = ""
        classes = predicting(image)
        for i in range(33):
            if classes[i] > 0.1:
                result += f"{print_letter(i)} - {classes[i] * 100}%\n"
        print(result)
        cv2.waitKey(0)

    return result


def start() -> None:
    results = files_predicting()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    start()
