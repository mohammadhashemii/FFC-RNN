import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from typing import List

import torch


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


def greedy_decoder(emission: torch.Tensor, blank, ds):
    """Given a sequence emission over labels, get the best path
    Args:
      emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
      blank: blank character
      ds: Dataset Class
    Returns:
      List[str]: The resulting transcript
    """
    indices = torch.argmax(emission, dim=-1)  # [num_seq,]
    indices = torch.unique_consecutive(indices, dim=-1)
    indices = [int(i) for i in indices if i != blank]
    # joined = "".join([self.labels[i] for i in indices])
    words = [x for x in ds.wv.num_to_word(torch.IntTensor(indices)) if x != '<unk>']
    joined = " ".join(words)

    return joined, indices


def ctc_decode(log_probs, blank, ds, training=True):
    if training:
        emission_log_probs = np.transpose(log_probs.detach().cpu().numpy(), (1, 0, 2))
    else:
        emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoded_list = []
    indices_list = []
    for emission_log_prob in emission_log_probs:
        decoded, indices = greedy_decoder(torch.Tensor(emission_log_prob), blank, ds)
        decoded_list.append(decoded)
        indices_list.append(indices)
    return decoded_list, indices_list
