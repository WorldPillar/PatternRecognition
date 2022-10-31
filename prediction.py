import numpy as np
import glob
import tensorflow as tf

val_dir = 'mytest/'


def print_letter(result):
    letters = "ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    return letters[result]


def predicting(path_to_image):
    image = tf.keras.preprocessing.image
    model = tf.keras.models.load_model('model.h5')

    img = image.load_img(path_to_image, target_size=(278, 278))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=1)
    result = int(np.argmax(classes))
    result = print_letter(result)
    print(result)


def start() -> None:
    for image in glob.glob('val/1/*'):
        predicting(image)
    # for folder in glob.glob(f'{val_dir}*'):
    #     print(folder)
    #     for image in glob.glob(f'{folder}/*'):
    #         predicting(image)


if __name__ == '__main__':
    start()
