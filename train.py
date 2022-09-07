import argparse
import os
import json

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from dataset import SadriDataset
from tqdm import tqdm
from models.FFCRnn import FFCRnn
from losses.CTC_loss import compute_ctc_loss
from utils import ctc_decode
import torch.nn.functional as F
from metric import HandwrittenRecognitionMetrics
from utils import visualize_samples

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', default='data', help='path to dataset directory')
parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
# training
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint')
parser.add_argument('--pretrained', default=False, action="store_true", help='use pretrained model')
parser.add_argument('--finetune', default=False, action="store_true", help='fine-tune the model with lr=1e-5')
parser.add_argument('--char_based', default=False, action="store_true", help='char based or word based vocabulary')
parser.add_argument('--feature_extractor', type=str, required=True, help='feature extractor name (e.g. ffc_resnet18)')
parser.add_argument('--use_attention', default=False, action="store_true", help='use attention layer')
parser.add_argument('--n_rnn', type=int, default=3, help='number of LSTM layers')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--imgH', type=int, default=32, help='input image height')
parser.add_argument('--imgW', type=int, default=256, help='input image width')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--learning_rate_decay', type=float, default=0.1, help='learning rate decay steps')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

if not os.path.exists(os.path.join(args.exp_dir, args.exp)):
    os.makedirs(os.path.join(args.exp_dir, args.exp))

# load data and create data loaders
img_size = (args.imgW, args.imgH)
train_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=True, char_based=args.char_based)
test_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=False, char_based=args.char_based)
#visualize_samples(train_dataset, SadriDataset.wv.num_to_word, random_img=True, n_samples=20)


loader_args = dict(batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=True)
train_loader = DataLoader(train_dataset, shuffle=False, **loader_args)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
n_train = len(train_dataset)

# create the model
# ffc_rnn = FFCRnn(image_height=args.imgH,
#                  nc=1,  # since the images are black and white
#                  nh=256,
#                  output_number=len(train_dataset.wv.word_vocab) + 1,
#                  n_rnn=4,
#                  leaky_relu=False,
#                  map_to_seq_hidden=512,
#                  feature_extractor=ffc_resnet18())

ffc_rnn = FFCRnn(nh=256,
                 output_number=len(train_dataset.wv.word_vocab) + 1,
                 n_rnn=args.n_rnn,
                 feature_extractor=args.feature_extractor,
                 use_attention=args.use_attention)

ffc_rnn.to(device=device)
if args.resume is not None:
    ffc_rnn.load_state_dict(torch.load(os.path.join(args.exp_dir, args.exp, args.resume)))
    print(f"Model weights {os.path.join(args.exp_dir, args.exp, args.resume)} loaded!")


lr = args.learning_rate

if args.pretrained: # freeze the model params
    print("Using pretrained model weights!")
    for param in ffc_rnn.parameters():
        param.requires_grad = False

    num_ftrs = ffc_rnn.fc.in_features
    ffc_rnn.fc = nn.Linear(num_ftrs, len(train_dataset.wv.word_vocab) + 1)  # unfreeze the last fc layer
    ffc_rnn.to(device=device)

if args.finetune:
    print("Fine-tuning mode!")
    lr = 1e-5   # reduce the chance of over-fitting

# set up optimizer, learning rate, etc.
optimizer = torch.optim.Adam(ffc_rnn.parameters(),
                             lr=lr)
scheduler = StepLR(optimizer, step_size=40, gamma=args.learning_rate_decay)

# Initialize logging
training_configs = dict(exp=args.exp, epochs=args.n_epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers, learning_rate=lr,
                        learning_rate_decay=args.learning_rate_decay,
                        device=device.type, feature_extractor=args.feature_extractor, n_rnn=args.n_rnn,
                        use_attention=args.use_attention)
json_path = os.path.join(args.exp_dir, args.exp, f"training_configs.json")
with open(json_path, "w") as jf:
    json.dump(training_configs, jf)
print(f"Training configs:\n {training_configs}")

# training
epoch_train_loss_list = []
epoch_val_loss_list = []
least_ser = None
least_wer = None
for epoch in range(args.n_epochs):
    ############## Training ##############
    ffc_rnn.train()
    wrong_cases = []
    train_metric = HandwrittenRecognitionMetrics()
    with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{args.n_epochs}", unit='img') as pbar:
        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)  # (batch, c=1, imgH, imgW)
            labels = batch['label'].to(device=device, dtype=torch.int)

            preds = ffc_rnn(images)
            log_probs = F.log_softmax(preds, dim=2)
            loss = compute_ctc_loss(log_probs, labels) / images.size()[0]
            epoch_train_loss_list.append(loss)

            ffc_rnn.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ffc_rnn.parameters(), 5)
            optimizer.step()
            pbar.update(images.shape[0])

            decoded_preds, indices_list = ctc_decode(log_probs.detach().clone(), 0, SadriDataset, char_based=args.char_based)
            target_lengths = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in labels])
            target_length_counter = 0
            for i in range(len(labels)):
                label_indices = labels[i]
                l = SadriDataset.wv.num_to_word(label_indices)
                words = [w for w in l if w != '<unk>']
                if args.char_based:
                    ground_truth_sentence = "".join(words)
                else:
                    ground_truth_sentence = " ".join(words)

                # update WER and SER metrics for one epoch
                train_metric.update_metric(decoded_preds[i], ground_truth_sentence, char_based=args.char_based)

        avg_train_loss = torch.mean(torch.tensor(epoch_train_loss_list))
        if not args.char_based:
            train_wer = train_metric.wer
            train_ser = train_metric.ser
        else:
            train_wer = train_metric.wer
            train_cer = train_metric.cer

        ############## validation ##############
        ffc_rnn.eval()
        wrong_cases = []
        val_metric = HandwrittenRecognitionMetrics()
        for batch in test_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.int)
            with torch.no_grad():
                preds = ffc_rnn(images)

                log_probs = F.log_softmax(preds, dim=2)
                val_loss = compute_ctc_loss(log_probs, labels).item() // images.size(0)
                epoch_val_loss_list.append(val_loss)

                decoded_preds, indices_list = ctc_decode(log_probs, 0, SadriDataset, char_based=args.char_based)
                # target_lengths = torch.IntTensor([len(t) for t in labels])
                target_lengths = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in labels])
                target_length_counter = 0
                for i in range(len(labels)):
                    label_indices = labels[i]
                    l = SadriDataset.wv.num_to_word(label_indices)
                    words = [w for w in l if w != '<unk>']
                    if args.char_based:
                        ground_truth_sentence = "".join(words)
                    else:
                        ground_truth_sentence = " ".join(words)

                    # update WER and SER metrics for one epoch
                    val_metric.update_metric(decoded_preds[i], ground_truth_sentence, char_based=args.char_based)

        avg_val_loss = torch.mean(torch.tensor(epoch_val_loss_list))

        if not args.char_based:
            val_wer = val_metric.wer
            val_ser = val_metric.ser
        else:
            val_wer = val_metric.wer
            val_cer = val_metric.cer

        # schedule the LR
        scheduler.step()

        # log losses and ... in a text file
        log_file_path = os.path.join(args.exp_dir, args.exp, f"training.log")
        with open(log_file_path, "a") as f:
            epoch_msg = f"Epoch: {epoch}"
            loss_msg = f"\nTrain CTC loss: {round(avg_train_loss.item(), 2)}, Val CTC loss:{round(avg_val_loss.item(), 2)}"
            if not args.char_based:
                metric_msg = f"Train SER: {round(100 * train_ser, 2)}\tTrain WER:{round(100 * train_wer, 2)}\tVal SER: {round(100 * val_ser, 2)}\tVal WER:{round(100 * val_wer, 2)}"
            else:
                metric_msg = f"Train CER: {round(100 * train_cer, 2)}\tTrain WER:{round(100 * train_wer, 2)}\tVal CER: {round(100 * val_cer, 2)}\tVal WER:{round(100 * val_wer, 2)}"
            print(f"{epoch_msg}\t{loss_msg}\n{metric_msg}\n")
            f.write(f"{epoch_msg}\t{loss_msg}\n{metric_msg}\n")
        f.close()

        # do checkpointing
        if not os.path.exists(os.path.join(args.exp_dir, args.exp, 'checkpoints')):
            os.makedirs(os.path.join(args.exp_dir, args.exp, 'checkpoints'))
        if not args.char_based:
            if least_ser is None or val_ser < least_ser:
                checkpoint_path = os.path.join(args.exp_dir, args.exp, 'checkpoints', f"checkpoint_best.pth")
                torch.save(ffc_rnn.state_dict(), checkpoint_path)
                least_ser = val_ser
        else:
            if least_wer is None or val_wer < least_wer:
                checkpoint_path = os.path.join(args.exp_dir, args.exp, 'checkpoints', f"checkpoint_best.pth")
                torch.save(ffc_rnn.state_dict(), checkpoint_path)
                least_wer = val_wer
        checkpoint_path = os.path.join(args.exp_dir, args.exp, 'checkpoints', f"checkpoint_last.pth")
        torch.save(ffc_rnn.state_dict(), checkpoint_path)
