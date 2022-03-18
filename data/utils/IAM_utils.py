import os
import xmltodict
import cv2
import random
random.seed(123)

def get_xml_filenames(uttlist_path: str):
    with open(uttlist_path, "r") as f:
        lines = [line.rstrip('\n')+'.xml' for line in f]

    return lines


def parse_xml_to_read_lines(xml_path: str):
    with open(xml_path, 'r', encoding='utf-8') as file:
        my_xml = file.read()

    # Use xmltodict to parse and convert the XML document
    my_dict = xmltodict.parse(my_xml)

    # Print the dictionary
    lines = my_dict['form']['handwritten-part']['line']
    lines_list = []
    for line in lines:
        words = line['word']
        line_label_list = []    # to use as the label of the entire line
        line_words_images_filename_list = []
        if type(words) is list:
            for word in words:
                line_label_list.append(word['@text'])
                line_words_images_filename_list.append(word['@id'] + '.png')
            #print(line_words_images_filename_list)
            #print(line_label_list)
        else:   # when a line contains only one word
            line_label_list.append(words['@text'])
            line_words_images_filename_list.append(words['@id'] + '.png')


        line_entry = {'line_words_images_filename_list': line_words_images_filename_list,
                      'line_label_list': line_label_list}
        lines_list.append(line_entry)

    return lines_list


def get_word_img_full_path(filename: str):
    level_one_dir = filename.split('-')[0]
    level_two_dir = filename.split('-')[0] + '-' + filename.split('-')[1]

    return os.path.join(level_one_dir, level_two_dir, filename)


def hconcat_resize(img_list):
    # take minimum height
    h_max = max(img.shape[0] for img in img_list)
    img_list_resize = []
    for idx, img in enumerate(img_list):
        white = [255, 255, 255]
        img_list_resize.append(cv2.copyMakeBorder(img, int((h_max - img.shape[0]) / 2), h_max - img.shape[0] - int((h_max - img.shape[0]) / 2),
                       random.randint(15, 20), random.randint(15, 20), cv2.BORDER_CONSTANT, value=white))
    # return final image
    return cv2.hconcat(img_list_resize)


def create_sentence_image(line_words_images_filename_list, words_base_path):
    word_img_list = []
    word_img_path_list = [os.path.join(words_base_path, get_word_img_full_path(fn))
                          for fn in line_words_images_filename_list]
    for word_img_path in word_img_path_list:
        word_img = cv2.imread(word_img_path)
        # concat word images horizontally
        word_img_list.append(word_img)

    sentence_img = hconcat_resize(word_img_list)

    return sentence_img


def generate_sample(images_dir, labels_dir, sample_info: dict, mode='sentence'):
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(labels_dir):
        os.mkdir(labels_dir)

    line_words_images_filename_list = sample_info['line_words_images_filename_list']
    line_label_list = sample_info['line_label_list']
    if mode == 'sentence':
        img = create_sentence_image(line_words_images_filename_list, words_base_path='../IAM/words')
        ground_truth = ' '.join(line_label_list)

        # save the created sample
        img_filename = '-'.join(line_words_images_filename_list[0].split('-')[:-1]) + '.png'
        cv2.imwrite(os.path.join(images_dir, img_filename), img)

        # save the ground truth as a text file
        label_filename = '-'.join(line_words_images_filename_list[0].split('-')[:-1]) + '.txt'
        with open(os.path.join(labels_dir, label_filename), 'w') as f:
            f.write(ground_truth)
        f.close()
    elif mode == 'word':
        for i, word_filename in enumerate(line_words_images_filename_list):
            img = cv2.imread(os.path.join('../IAM/words', get_word_img_full_path(word_filename)))
            ground_truth = line_label_list[i]
            # save the created sample
            img_filename = line_words_images_filename_list[i]
            try:
                cv2.imwrite(os.path.join(images_dir, img_filename), img)
                # save the ground truth as a text file
                label_filename = line_words_images_filename_list[i].split('.')[0] + '.txt'
                with open(os.path.join(labels_dir, label_filename), 'w') as f:
                    f.write(ground_truth)
                f.close()
            except cv2.error: # could not able to read the image
                print(f"Could not save image: {word_filename}")


def split_train_test_samples(train_uttlist_path, test_uttlist_path):
    # train files
    with open(train_uttlist_path, "r") as f:
        xml_filenames = [line.rstrip('\n') for line in f]
    total_file_names = set()
    for txf in xml_filenames:
        page_lines_list = [x for x in os.listdir('../IAM_dataset/images') if x.startswith(txf)]
        total_file_names.update(page_lines_list)
    with open('../IAM_dataset/train.txt', 'w') as train_file:
        for fn in total_file_names:
            train_file.write(f"{fn}\n")

    train_file = open('../IAM_dataset/train.txt', 'r')
    print("Number of train samples: {}".format(len(train_file.readlines())))
    train_file.close()

    # test files
    with open(test_uttlist_path, "r") as f:
        xml_filenames = [line.rstrip('\n') for line in f]
    total_file_names = set()
    for txf in xml_filenames:
        page_lines_list = [x for x in os.listdir('../IAM_dataset/images') if x.startswith(txf)]
        total_file_names.update(page_lines_list)
    with open('../IAM_dataset/test.txt', 'w') as test_file:
        for fn in total_file_names:
            test_file.write(f"{fn}\n")

    test_file = open('../IAM_dataset/test.txt', 'r')
    print("Number of test samples: {}".format(len(test_file.readlines())))
    test_file.close()


if __name__ == "__main__":

    print('Reading XML files...')
    xml_filenames = get_xml_filenames(uttlist_path='../IAM/splits/train_val_test.uttlist')
    print('Generating samples...')
    print('=====================')
    print('The following samples are removed from the final dataset:')
    for xf in xml_filenames:
        line_lists = parse_xml_to_read_lines(xml_path=os.path.join('../IAM/xml/', xf))
        for line_dict in line_lists:

            # print some info for debugging purposes
            try:
                generate_sample(images_dir='../IAM_dataset/images', labels_dir='../IAM_dataset/labels',
                                sample_info=line_dict, mode='word')
            except AttributeError:
                print(f"{line_dict['line_words_images_filename_list'][0]}: {' '.join(line_dict['line_label_list'])} ")
    

    split_train_test_samples(train_uttlist_path='../IAM/splits/train_val.uttlist',
                             test_uttlist_path='../IAM/splits/test.uttlist')