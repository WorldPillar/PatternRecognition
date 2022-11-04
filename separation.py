import shutil
import os
import glob

data_dir = 'modified_images'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
# Часть набора данных для тестирования
test_data_portion = 0.20
# Часть набора данных для проверки
val_data_portion = 0.10
# Количество элементов данных в одном классе
nb_images = 666
images_list = []


def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    for i in range(33):
        letter = '{0:06b}'.format(i)
        os.makedirs(os.path.join(dir_name, f'{letter}'))


def get_images_list():
    x = 0
    for folder in glob.glob(f'{data_dir}/*'):
        images_list.append([])
        y = 0
        for image in glob.glob(f'{folder}/*'):
            images_list[x].append(image.split('\\')[2])
            y += 1
        x += 1


def copy_images(start_index, end_index, dest_dir):
    for i in range(start_index, end_index):
        x = 0
        for folder in glob.glob(f'{data_dir}/*'):
            letter = folder.split('\\')[1]
            shutil.copy2(os.path.join(folder, images_list[x][i]),
                         os.path.join(dest_dir, f'{letter}'))
            x += 1


def start() -> None:
    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)

    get_images_list()

    start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
    start_test_data_idx = int(nb_images * (1 - test_data_portion))

    copy_images(0, start_val_data_idx, train_dir)
    copy_images(start_val_data_idx, start_test_data_idx, val_dir)
    copy_images(start_test_data_idx, nb_images, test_dir)


if __name__ == '__main__':
    start()
