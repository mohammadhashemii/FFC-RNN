# FFC-RNN

This is a PyTorch Implementation of FFC-RNN (Fast Fourier Convolution + RNN) with CTC loss for Handwritten Recognition.

**_Note_**: This work is highly inspired by [Fast Fourier Convolution](https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf) paper published in NIPS.

## Dataset
<p align="center">
  <img src="https://github.com/mohammadhashemii/FFC-RNN/blob/main/img/9980000000_2.jpg">	
</p>
This project aimed to perform handwritten recognition on Persian numbers written in text. In this way, we used specific parts(those containing numbers) of [Sadri Dataset](https://users.encs.concordia.ca/~j_sadri/PersianDatabase.htm). This work has also been tested on [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). In order to run the code and train the model on your own data, you must create a directory similar to the one shown below and manipulate the code written in `dataset.py`:

```
data/images/ : contains images
data/labels/ : contains labels in text files
data/train.txt : contains the filenames for train set
data/test.txt : contains the filenames for test set
```

## Sample Usage for Training the Model

```
!python train.py \
--exp 01 \
--learning_rate_decay 1 \
--learning_rate 0.0001 \
--char_based \
--exp_dir /path/to/experiments/directory \
--resume /path/to/saved/weights.pth \
--feature_extractor cnn \
--n_rnn 2 \
--imgW 128 \
--batch_size 64 \
--n_epochs 100 \
--data_root data \
```

## Question

Of course, the code is not well-documented yet, and we have not completed a comprehensive guideline for using the code. So any questions or issues are welcome.
