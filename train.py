import argparse
import os
import json

import torch
from torch.utils.data import DataLoader
from dataset import SadriDataset
from tqdm import tqdm
from models.FFCRnn import FFCRnn
from losses.CTC_loss import compute_ctc_loss
from torchsummary import summary

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', default='data', help='path to dataset directory')
parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
# training
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
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
ffc_rnn = FFCRnn(image_height=args.imgH,
                 nc=1,  # since the images are black and white
                 nh=256,
                 output_number=len(train_dataset.wv.word_vocab) + 3,
                 n_rnn=4,
                 leaky_relu=False)

ffc_rnn.to(device=device)

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

#model_summary = summary(ffc_rnn, (1, args.imgH, args.imgW), batch_size=-1, device='cuda')
#model_summary_path = os.path.join(args.exp_dir, args.exp, f"model_summary.txt")
#with open(model_summary, "w") as f:
#    f.write(model_summary)
#    f.close()

# training
epoch_train_loss_list = []
epoch_val_loss_list = []
for epoch in range(args.n_epochs):
    ffc_rnn.train()
    with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{args.n_epochs}", unit='img') as pbar:
        for batch in train_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)  # (batch, c=1, imgH, imgW)
            labels = batch['label'].to(device=device, dtype=torch.int)

            preds = ffc_rnn(images)

            loss = compute_ctc_loss(preds, labels)
            epoch_train_loss_list.append(loss)
            ffc_rnn.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(images.shape[0])
        avg_train_loss = torch.mean(torch.tensor(epoch_train_loss_list))

        # validation
        ffc_rnn.eval()
        for batch in test_loader:
            images = batch['image'].to(device=device, dtype=torch.float32)
            labels = batch['label'].to(device=device, dtype=torch.int)
            with torch.no_grad():
                preds = ffc_rnn(images)
                val_loss = compute_ctc_loss(preds, labels)
                epoch_val_loss_list.append(val_loss)
        avg_val_loss = torch.mean(torch.tensor(epoch_val_loss_list))

        # log losses and ... in a text file
        log_file_path = os.path.join(args.exp_dir, args.exp, f"training.log")
        with open(log_file_path, "a") as f:
            loss_msg = f"Train loss: {avg_train_loss}, Test loss:{avg_val_loss}\n"
            # TODO: compute accuracy
            print(loss_msg)
            f.write(loss_msg)
        f.close()

        # do checkpointing
        if not os.path.exists(os.path.join(args.exp_dir, args.exp, 'checkpoints')):
            os.makedirs(os.path.join(args.exp_dir, args.exp, 'checkpoints'))
        checkpoint_path = os.path.join(args.exp_dir, args.exp, 'checkpoints', f"ffc_rnn_epoch_{epoch}")
        torch.save(ffc_rnn.state_dict(), checkpoint_path)
