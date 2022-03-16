import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np
from bidi.algorithm import get_display
from arabic_reshaper import reshape
from typing import List
import json
from scipy.special import logsumexp

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


def preprocess_image(img, img_size, apply_augmentation=False):
    img = distortion_free_resize(img, img_size=img_size)

    if apply_augmentation:
        img = torchvision.transforms.Compose([
            torchvision.transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 0.3), value=255),
            torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ])(img)

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


def _reconstruct(labels, blank=0):
    new_labels = []
    # merge same labels
    previous = None
    for label in labels:
        if label != previous:
            new_labels.append(label)
            previous = label
    # delete blank
    new_labels = [new_label for new_label in new_labels if new_label != blank]

    return new_labels


def beam_search_decode(emission_log_prob, blank, ds, char_based=False, **kwargs):
    NINF = -1 * float('inf')
    DEFAULT_EMISSION_THRESHOLD = 0.01

    # beam_size = kwargs['beam_size']
    # emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    beam_size = 10
    emission_threshold = np.log(DEFAULT_EMISSION_THRESHOLD)

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                # log(p1 * p2) = log_p1 + log_p2
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        # sorted by accumulated_log_prob
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    # sum up beams to produce labels
    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        # log(p1 + p2) = logsumexp([log_p1, log_p2])
        total_accu_log_prob[labels] = logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob)
                    for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    labels = labels_beams[0][0]

    words = [x for x in ds.wv.num_to_word(torch.IntTensor(labels)) if x != '<unk>']
    if char_based:
        joined = "".join(words)
    else:
        joined = " ".join(words)
    
    return joined, labels


def greedy_decoder(emission: torch.Tensor, blank, ds, char_based=False):
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
    indices = [i for i in indices if i != blank]
    # joined = "".join([self.labels[i] for i in indices])
    words = [x for x in ds.wv.num_to_word(torch.IntTensor(indices)) if x != '<unk>']
    if char_based:
        joined = "".join(words)
    else:
        joined = " ".join(words)

    return joined, indices


def ctc_decode(log_probs, blank, ds, beam_decode=False, char_based=False):
    emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
    # size of emission_log_probs: (batch, length, class)

    decoded_list = []
    indices_list = []
    for emission_log_prob in emission_log_probs:
        if beam_decode:
            decoded, indices = beam_search_decode(torch.Tensor(emission_log_prob), blank, ds, char_based=char_based)
        else:
            decoded, indices = greedy_decoder(torch.Tensor(emission_log_prob), blank, ds, char_based=char_based)
        decoded_list.append(decoded)
        indices_list.append(indices)
    return decoded_list, indices_list


def save_vocab_dict(dataset, json_path):
    '''
    save the vocab dictionary as json file
    '''
    d = {}
    for i in range(1, len(dataset.wv.le.vocab)):
        word = dataset.wv.le.vocab[i]
        d[dataset.wv.le.encode(word).item()] = word

    with open(json_path, "w", encoding='utf8') as jf:
        json.dump(d, jf, ensure_ascii=False)

    print(f"Vocab dictionary saved at {json_path}")
