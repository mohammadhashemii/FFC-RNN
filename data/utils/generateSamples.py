'''
This script is for generating samples using combining words images
which exist in /words directory.
'''

import os
import random
import shutil

import num2word
import randomNumGenerator
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd


def hconcat_resize(img_list):
    # take minimum height
    h_max = max(img.shape[0] for img in img_list)
    white = [255, 255, 255]
    img_list_resize = [
        cv2.copyMakeBorder(img, int((h_max - img.shape[0]) / 2), h_max - img.shape[0] - int((h_max - img.shape[0]) / 2),
                           0, 0, cv2.BORDER_CONSTANT, value=white)
        for img in img_list]

    # return final image
    return cv2.hconcat(img_list_resize)


def create_num_image(num_string, words_base_path='../data/train_words/'):
    words = num_string.split(" ")
    word_img_list = []
    for word in words:

        image_dir_path = words_base_path + str(word) + "/"

        all_images_in_directory = os.listdir(image_dir_path)
        if ".DS_Store" in all_images_in_directory: all_images_in_directory.remove(".DS_Store")

        # choose a random word image from the related directory
        random_word_image_path = random.choice(all_images_in_directory)
        word_img = cv2.imread(image_dir_path + random_word_image_path)
        # concat word images horizontally
        word_img_list.insert(0, word_img)

    num_string_image = hconcat_resize(word_img_list)

    return num_string_image

def read_nums_from_excel(path='../data/CourtesyValues.xlsx'):

    df = pd.read_excel(path)
    df = df.astype('int')
    df['TUTAR'] = df['TUTAR'] * 10000
    nums = df['TUTAR'].values

    return nums

def generate_images(images_dir, labels_dir):
    # create the /images and /labels directory in case they do not exist
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    # fetch the random numbers generated via randomNumGenerator.py script
    #random_numbers = read_nums_from_excel(path='../data/CourtesyValues.xlsx')
    random_numbers = randomNumGenerator.generate_random_num()
    random.shuffle(random_numbers)

    train_samples, test_samples = train_test_split(random_numbers, test_size=0.1)


    print('Creating the images...')
    idx = 0
    repeated_imgs = 0
    # create the images for train set
    for num in train_samples:

        num = int(num)
        print('{0}/{1}: {2}'.format(idx, len(random_numbers), num))
        num_string = num2word.convert(num)
        num_string += " ریال"
        num_img = create_num_image(num_string, words_base_path='../OurDataset/train_words/')
        # convert the raw image to a binary one
        (threshold, num_img) = cv2.threshold(num_img, 127, 255, cv2.THRESH_BINARY)

        # save the image and label into /images and /labels directory respectively
        img_path = images_dir + str(num)
        if os.path.exists(img_path + '.jpg'):
            img_path += "_2"
            while (os.path.exists(img_path + '.jpg')):
                img_path = img_path.split('_')[0] + '_' + str(int(img_path.split('_')[1])+1)

            repeated_imgs += 1
        cv2.imwrite(img_path + '.jpg', num_img)

        f = open(labels_dir + img_path.split('/')[-1] + '.txt', 'w')
        f.write(num_string)
        f.close()

        idx += 1

    # create the images for test set
    for num in test_samples:

        num = int(num)
        print('{0}/{1}: {2}'.format(idx, len(random_numbers), num))
        num_string = num2word.convert(num)
        num_string += " ریال"
        num_img = create_num_image(num_string, words_base_path='../OurDataset/test_words/')
        # convert the raw image to a binary one
        (threshold, num_img) = cv2.threshold(num_img, 127, 255, cv2.THRESH_BINARY)

        # save the image and label into /images and /labels directory respectively
        img_path = images_dir + str(num)
        if os.path.exists(img_path + '.jpg'):
            img_path += "_2"
            while (os.path.exists(img_path + '.jpg')):
                img_path = img_path.split('_')[0] + '_' + str(int(img_path.split('_')[1]) + 1)

            repeated_imgs += 1
        cv2.imwrite(img_path + '.jpg', num_img)

        f = open(labels_dir + img_path.split('/')[-1] + '.txt', 'w')
        f.write(num_string)
        f.close()

        idx += 1

    print("Number of repeated images:{}".format(repeated_imgs))

    print('Creating train.txt and test.txt list files...')
    f = open('../OurDataset/train.txt', 'w')
    for sample in train_samples:
        f.write(str(sample) + '.jpg' + '\n')

    f = open('../OurDataset/test.txt', 'w')
    for sample in test_samples:
        f.write(str(sample) + '.jpg' + '\n')
    f.close()


def split_words_to_train_test(test_size=0.1, train_path="../data/train_words/",  test_path="../data/test_words/"):
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    for word in os.listdir('../data/words/'):
        imgs = os.listdir(os.path.join('../data/words', word))
        train_words_imgs, test_words_imgs = train_test_split(imgs, test_size=test_size)

        # create a word directory in train_words
        os.mkdir(os.path.join(train_path, word))
        # create a word directory in test_words
        os.mkdir(os.path.join(test_path, word))

        # copy images
        print('Copying train images for: {}...'.format(word))
        for img in train_words_imgs:
            original = os.path.join('../data/words/' + word, img)
            destination = os.path.join(train_path + word, img)
            shutil.copyfile(original, destination)

        print('Copying test images for: {}...'.format(word))
        for img in test_words_imgs:
            original = os.path.join('../data/words/' + word, img)
            destination = os.path.join(test_path + word, img)
            shutil.copyfile(original, destination)



if __name__ == '__main__':
    generate_images(images_dir='../OurDataset/images/', labels_dir='../OurDataset/labels/')
    #split_words_to_train_test(test_size=0.1)
    #print(num2word.convert(1111123))