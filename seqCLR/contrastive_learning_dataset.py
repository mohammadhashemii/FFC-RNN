import os
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from seqCLR.transformations import *
import copy
import matplotlib.pyplot as plt
import numpy as np


def distortion_free_resize(img, img_size):
    w, h = img_size
    old_w, old_h = img.shape[2], img.shape[1]
    new_size = h if old_w >= old_h else w

    img = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(new_size),

    ])(img)

    # check the amount of padding needed to be done
    pad_height = h - img.shape[1]
    pad_width = w - img.shape[2]

    # only necessary if you want to do same amount of padding on both sides
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    img = torchvision.transforms.Compose([
        torchvision.transforms.Pad([pad_width_left, pad_height_top,
                                    pad_width_right, pad_height_bottom]),

    ])(img)

    return img


class PersianHandwrittenStringsDataset(Dataset):
    def __init__(self, data_dir, name, transform=None):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, 'images')
        self.name = name
        self.filenames = os.listdir(self.images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_filename = self.filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        img = torchvision.io.read_image(str(img_path))
        img = distortion_free_resize(img, img_size=(256, 32))
        img = transforms.Compose([transforms.ToPILImage(),
                                  ])(img)
        img = self.transform(img)

        return img


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


class ContrastiveLearningDataset:
    def __init__(self, data_dir='data', img_size=(256, 32)):
        self.data_dir = data_dir
        self.img_size = img_size

    @staticmethod
    def get_seqclr_pipeline_transform(img_size):
        linear_contrast = transforms.ColorJitter(0, 0.8, 0, 0)
        data_transforms = transforms.Compose([transforms.GaussianBlur(kernel_size=int(0.1 * img_size[1])),
                                              transforms.RandomApply([linear_contrast], p=0.8),
                                              transforms.RandomAdjustSharpness(2),
                                              transforms.RandomAffine(degrees=(-3, 3), shear=(0.8, 1), fill=255),
                                              transforms.ToTensor(),
                                              AddGaussianNoise(0, 0.05)
                                              ])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'SadriDataset': lambda: PersianHandwrittenStringsDataset(data_dir=self.data_dir,
                                                                     name=name,
                                                                     transform=ContrastiveLearningViewGenerator(
                                                                         self.get_seqclr_pipeline_transform(
                                                                             img_size=self.img_size),
                                                                         n_views=n_views)),
            'OurDataset': lambda: PersianHandwrittenStringsDataset(data_dir=self.data_dir,
                                                                   name=name,
                                                                   transform=ContrastiveLearningViewGenerator(
                                                                       self.get_seqclr_pipeline_transform(
                                                                           img_size=self.img_size),
                                                                       n_views=n_views
                                                                   ))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise ValueError
        else:
            return dataset_fn()


def visualize_samples(dataset, n_samples=8, cols=4, random_img=False):
    dataset = copy.deepcopy(dataset)
    rows = n_samples // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(n_samples):
        if random_img:
            idx = np.random.randint(1, len(dataset))
        else:
            idx = i
        sample = dataset[idx]
        img = sample[0]
        ax.ravel()[i].imshow(img.permute(1, 2, 0), cmap='gray')
        ax.ravel()[i].set_axis_off()

    plt.tight_layout(pad=1)
    plt.show()

# c = PersianHandwrittenStringsDataset(data_dir="data", name="OurDataset")
# dataset = ContrastiveLearningDataset(data_dir="data", img_size=(256, 32))
# train_dataset = dataset.get_dataset('SadriDataset', 1)
# print(train_dataset[0][0].shape)
# visualize_samples(train_dataset, random_img=True, n_samples=20)