import numpy as np
import glob
import tensorflow as tf

class_count = 33
val_dir = 'val/'
offset = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 30, 31, 32, 4, 5,
          6, 7, 8, 9]


def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]


def predicting(path_to_image):
    image = tf.keras.preprocessing.image
    model = tf.keras.models.load_model('model.h5')

    img = image.load_img(path_to_image, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    result = int(np.argmax(classes))
    # result = print_letter(offset[result])
    return offset[result]


def class_accuracy(right, all):
    print(right / all)


def start() -> None:
    right = []
    class_numb = 0
    for folder in glob.glob(f'{val_dir}*'):
        print(f"Expected: {print_letter(offset[class_numb])} From {folder}")
        right.append(0)
        count = 0
        for image in glob.glob(f'{folder}/*'):
            result = predicting(image)
            count += 1
            if result == offset[class_numb]:
                right[class_numb] += 1
        class_accuracy(right[class_numb], count)
        class_numb += 1
    print(np.sum(right) / len(right))


if __name__ == '__main__':
    start()
