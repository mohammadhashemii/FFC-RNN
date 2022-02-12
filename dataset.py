import os
import torchvision
from torch.utils.data import Dataset
from torchnlp.encoders import LabelEncoder
import torch.nn.functional as F

from utils import preprocess_image

class SadriDataset(Dataset):
    def __init__(self, root_dir='data', img_size=(256, 32), is_training_set=True):
        '''
        It is expected to have data directory with the following structure:
        data/images/ : contains images
        data/labels/ : contains labels in text files
        data/train.txt : contains the filenames for train set
        data/test.txt : contains the filenames for test set
        '''
        self.is_training_set = is_training_set
        self.root = root_dir
        self.images_dir = os.path.join(self.root, 'images')
        self.labels_dir = os.path.join(self.root, 'labels')
        if is_training_set:
            self.filenames_path = os.path.join(self.root, 'train.txt')
        else:
            self.filenames_path = os.path.join(self.root, 'test.txt')
        self.paths_and_labels = self._get_paths_and_labels()
        if is_training_set:
            self.max_len, self.word_vocab = self._create_words_vocab(self.paths_and_labels['labels_list'])
            self.wv = WordVocabulary(word_vocab=self.word_vocab, max_len = self.max_len)
        self.img_w, self.img_h = img_size
        self.num_samples = len(self.paths_and_labels['labels_list'])


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = self.paths_and_labels['img_paths'][idx]
        img = torchvision.io.read_image(img_path)
        label = self.paths_and_labels['labels_list'][idx]
        sample = self._preprocess_image_and_label(img, label)

        return sample   # image, label

    def _read_filenames(self) -> list:
        with open(self.filenames_path) as f:
            filenames_list = f.read().splitlines()

        return filenames_list

    def _get_images_path_and_labels(self, filenames_list: list):
        img_path_list = []
        labels_list = []
        for fn in filenames_list:
            img_path_list.append(os.path.join(self.images_dir, fn))
            label_filename = os.path.join(self.labels_dir, fn.split('.')[0] + '.txt')
            with open(label_filename) as f:
                labels_list.append(f.read())

        return img_path_list, labels_list

    def _get_paths_and_labels(self) -> dict:
        filenames_list = self._read_filenames()
        img_paths_list, labels_list = self._get_images_path_and_labels(filenames_list)

        print(f"{len(labels_list)} images found for {'train' if self.is_training_set else 'test'} set!")
        return {'img_paths': img_paths_list,
                'labels_list': labels_list}

    def _create_words_vocab(self, labels: list):
        print("Creating word vocabulary...")
        word_vocab = set()
        max_len = 0 # longest label size
        for l in labels:
            l = l.split(' ')
            for word in l:
                word_vocab.add(word)
            max_len = max(max_len, len(l))

        print(f"Word vocabulary size: {len(word_vocab)}")
        print(f"Maximum label size: {max_len}")

        return max_len, word_vocab


    def _preprocess_image_and_label(self, img, label):
        img = preprocess_image(img=img, img_size=(self.img_w, self.img_h))
        label = self.wv.word_to_num(label=label)

        return {'image': img, 'label': label}

class WordVocabulary:
    def __init__(self, word_vocab: set, max_len: int):
        self.word_vocab = word_vocab
        self.padding_token = 99
        self.max_len = max_len
        self.le = LabelEncoder(list(self.word_vocab),)

    def word_to_num(self, label):
        label = label.split(' ')
        enl = self.le.batch_encode(label)
        length = enl.shape[0]
        pad_amount = self.max_len - length
        padding_token = 0
        label = F.pad(enl, (0, pad_amount), "constant", padding_token)

        return label

    def num_to_word(self, indices):
        return self.le.batch_decode(indices)

