import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape


def distortion_free_resize(img, img_size):
    w, h = img_size
    old_w, old_h = img.shape[2], img.shape[1]
    new_size = h if old_w >= old_h else w

    img = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(new_size)
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

def preprocess_image(img, img_size):
    img = distortion_free_resize(img, img_size=img_size)
    img = img / 255.

    return img

def visualize_samples(dataset, num_to_word, n_samples=8, cols=4, random_img=False):
    dataset = copy.deepcopy(dataset)
    rows = n_samples // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(n_samples):
        if random_img:
            idx = np.random.randint(1, len(dataset))
        else:
            idx = i
        sample = dataset[idx]
        img, label = sample['image'], sample['label']
        label = num_to_word(label)
        label = list(filter(('<unk>').__ne__, label))
        label = ' '.join(label)
        label = get_display(reshape(label))
        ax.ravel()[i].imshow(img.permute(1, 2, 0), cmap='gray')
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(label)

    plt.tight_layout(pad=1)
    plt.show()