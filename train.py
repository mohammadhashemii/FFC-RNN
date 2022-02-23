import argparse
import os
import json

import torch
from torch.utils.data import DataLoader
from dataset import SadriDataset
from tqdm import tqdm
from models.FFCRnn import FFCRnn
from models.FFCResnet import *
from losses.CTC_loss import compute_ctc_loss
from torchsummary import summary
from utils import ctc_decode, save_vocab_dict
import torch.nn.functional as F

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', default='data', help='path to dataset directory')
parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
# training
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--imgH', type=int, default=32, help='input image height')
parser.add_argument('--imgW', type=int, default=256, help='input image width')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

if not os.path.exists(os.path.join(args.exp_dir, args.exp)):
    os.makedirs(os.path.join(args.exp_dir, args.exp))

# load data and create data loaders
img_size = (args.imgW, args.imgH)
train_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=True)
test_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=False)

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
                 n_rnn=2,
                 feature_extractor="ffc_resnet18")

ffc_rnn.to(device=device)
if args.resume is not None:
    ffc_rnn.load_state_dict(torch.load(os.path.join(args.exp_dir, args.exp, args.resume)))
    print(f"Model weights {os.path.join(args.exp_dir, args.exp, args.resume)} loaded!")

# set up optimizer, learning rate, etc.
optimizer = torch.optim.Adam(ffc_rnn.parameters(),
                             lr=args.learning_rate)

# Initialize logging
training_configs = dict(exp=args.exp, epochs=args.n_epochs,
                        batch_size=args.batch_size, learning_rate=args.learning_rate,
                        device=device.type)
json_path = os.path.join(args.exp_dir, args.exp, f"training_configs.json")
with open(json_path, "w") as jf:
    json.dump(training_configs, jf)
print(f"Training configs:\n {training_configs}")

# model_summary = summary(ffc_rnn, (1, args.imgH, args.imgW), batch_size=-1, device='cuda')
# model_summary_path = os.path.join(args.exp_dir, args.exp, f"model_summary.txt")
# with open(model_summary, "w") as f:
#    f.write(model_summary)
#    f.close()

# training
epoch_train_loss_list = []
epoch_val_loss_list = []
least_loss = None
for epoch in range(args.n_epochs):
    ############## Training ##############
    ffc_rnn.train()
    tot_correct = 0
    wrong_cases = []
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

            decoded_preds, indices_list = ctc_decode(log_probs.detach().clone(), 0, SadriDataset)
            target_lengths = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in labels])
            target_length_counter = 0
            for i in range(len(labels)):
                label_indices = labels[i]
                l = SadriDataset.wv.num_to_word(label_indices)
                words = [w for w in l if w != '<unk>']
                ground_truth_sentence = " ".join(words)
                if ground_truth_sentence == decoded_preds[i]:
                    tot_correct += 1
                else:
                    wrong_cases.append((ground_truth_sentence, decoded_preds[i]))

        avg_train_loss = torch.mean(torch.tensor(epoch_train_loss_list))
        train_accuracy = tot_correct / train_dataset.num_samples

        ############## validation ##############
        ffc_rnn.eval()
        tot_correct = 0
        wrong_cases = []
        for batch in test_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.int)
            with torch.no_grad():
                preds = ffc_rnn(images)

                log_probs = F.log_softmax(preds, dim=2)
                val_loss = compute_ctc_loss(log_probs, labels).item() // images.size(0)
                epoch_val_loss_list.append(val_loss)

                decoded_preds, indices_list = ctc_decode(log_probs, 0, SadriDataset)
                # target_lengths = torch.IntTensor([len(t) for t in labels])
                target_lengths = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in labels])
                target_length_counter = 0
                for i in range(len(labels)):
                    label_indices = labels[i]
                    l = SadriDataset.wv.num_to_word(label_indices)
                    words = [w for w in l if w != '<unk>']
                    ground_truth_sentence = " ".join(words)
                    if ground_truth_sentence == decoded_preds[i]:
                        tot_correct += 1
                    else:
                        wrong_cases.append((ground_truth_sentence, decoded_preds[i]))

        avg_val_loss = torch.mean(torch.tensor(epoch_val_loss_list))
        val_accuracy = tot_correct / test_dataset.num_samples

        # log losses and ... in a text file
        log_file_path = os.path.join(args.exp_dir, args.exp, f"training.log")
        with open(log_file_path, "a") as f:
            msg = f"Train loss: {avg_train_loss}, Train acc:{train_accuracy}, Test loss:{avg_val_loss}, Val acc:{val_accuracy}\n"
            print(msg)
            f.write(msg)
        f.close()

        # do checkpointing
        if not os.path.exists(os.path.join(args.exp_dir, args.exp, 'checkpoints')):
            os.makedirs(os.path.join(args.exp_dir, args.exp, 'checkpoints'))
        if least_loss is None or avg_val_loss < least_loss:
            checkpoint_path = os.path.join(args.exp_dir, args.exp, 'checkpoints', f"checkpoint_best")
            torch.save(ffc_rnn.state_dict(), checkpoint_path)
            least_loss = avg_val_loss
