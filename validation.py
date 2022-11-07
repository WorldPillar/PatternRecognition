import numpy as np
import glob
import os
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('model.h5')
val_dir = 'val/'
image_width = 32
image_height = 32


def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]


def predicting(path_to_image):
    image = keras.preprocessing.image

    img = image.load_img(path_to_image, target_size=(image_width, image_height), color_mode="grayscale")
    x = image.img_to_array(img).astype(np.uint8) / 255.
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model(images)
    result = int(np.argmax(classes))
    return result


def files_predicting():
    result_list = []
    for folder in glob.glob(f'{val_dir}*'):
        print(folder)
        result_letters = []
        for image in glob.glob(f'{folder}/*'):
            result_letters.append(predicting(image))
        result_list.append(result_letters)
    return result_list


def letter_accuracy(prediction):
    accuracy = []
    x = 0
    for letters in prediction:
        accuracy.append(0)
        for letter in letters:
            if letter == x:
                accuracy[x] += 1
        accuracy[x] /= len(letters)
        x += 1
    return accuracy


def start() -> None:
    results = files_predicting()
    accuracy = letter_accuracy(results)

    print(accuracy)
    print(sum(accuracy)/len(accuracy))


if __name__ == '__main__':
    start()
