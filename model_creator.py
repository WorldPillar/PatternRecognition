from tensorflow import keras
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_width = 32
image_height = 32

ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
TRAINING_DIR = "train/"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
train_ds = train_datagen.flow_from_directory(TRAINING_DIR,
                                             batch_size=40,
                                             class_mode='binary',
                                             color_mode="grayscale",
                                             target_size=(image_width, image_height))

VALIDATION_DIR = "test/"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)
validation_ds = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                       batch_size=40,
                                                       class_mode='binary',
                                                       color_mode="grayscale",
                                                       target_size=(image_width, image_height))


def build_model():
    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu',
                            input_shape=(image_width, image_height, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(33, activation='softmax')
    ])

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def start():
    model = build_model()
    model.summary()
    history = model.fit(train_ds,
                        validation_data=validation_ds,
                        epochs=60,
                        verbose=1)
    model.save('model.h5')


if __name__ == '__main__':
    start()

