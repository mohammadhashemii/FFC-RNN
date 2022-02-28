import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
from dataset import SadriDataset
from models.FFCRnn import FFCRnn
from models.FFCResnet import *
from losses.CTC_loss import compute_ctc_loss
from utils import ctc_decode
import torch.nn.functional as F
from metric import HandwrittenRecognitionMetrics

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--model_state_path', required=True, help='path to model weights,'
                                                              ' note that it must be compatible with model in the code')
parser.add_argument('--data_root', default='data', help='path to dataset directory')
parser.add_argument('--inference_root', default='inference', help='path to saved results directory')
# testing
parser.add_argument('--feature_extractor', type=str, required=True, help='feature extractor name (e.g. ffc_resnet18)')
parser.add_argument('--use_attention', type=int, required=True, help='use attention layer')
parser.add_argument('--n_rnn', type=int, default=3, help='number of LSTM layers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--imgH', type=int, default=32, help='input image height')
parser.add_argument('--imgW', type=int, default=256, help='input image width')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not torch.cuda.is_available():
    print("WARNING: You are not using gpu!")

t = time.localtime()
current_time = time.strftime("%H_%M_%S", t)
inference_dir = os.path.join(args.inference_root, current_time)
if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)

# load test data and create its data loader
img_size = (args.imgW, args.imgH)
train_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=True)
test_dataset = SadriDataset(root_dir=args.data_root, img_size=img_size, is_training_set=False)

loader_args = dict(batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   pin_memory=True)

test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

# load the model
ffc_rnn = FFCRnn(nh=256,
                 output_number=len(train_dataset.wv.word_vocab) + 1,
                 n_rnn=args.n_rnn,
                 feature_extractor=args.feature_extractor,
                 use_attention=bool(args.use_attention))

ffc_rnn.to(device=device)
# load the weights
ffc_rnn.load_state_dict(torch.load(args.model_state_path))
print(f"Model weights ({args.model_state_path}) loaded!")

# log file
log_file_path = os.path.join(inference_dir, f"wrong_predicted_samples.txt")
log_file = open(log_file_path, "a")

# testing
ffc_rnn.eval()
wrong_cases = []
val_loss_list = []
test_metric = HandwrittenRecognitionMetrics()
for batch in test_loader:
    images = batch['image'].to(device=device, dtype=torch.float32)
    labels = batch['label'].to(device=device, dtype=torch.int)
    with torch.no_grad():
        preds = ffc_rnn(images)

        log_probs = F.log_softmax(preds, dim=2)
        val_loss = compute_ctc_loss(log_probs, labels).item() // images.size(0)
        val_loss_list.append(val_loss)

        decoded_preds, indices_list = ctc_decode(log_probs, 0, SadriDataset)

        target_lengths = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in labels])
        target_length_counter = 0
        for i in range(len(labels)):
            label_indices = labels[i]
            # label_indices = [i for i in label_indices if i != 0]
            l = SadriDataset.wv.num_to_word(label_indices)
            words = [w for w in l if w != '<unk>']
            ground_truth_sentence = " ".join(words)

            # update WER and SER metrics for one epoch
            test_metric.update_metric(decoded_preds[i], ground_truth_sentence)

            if ground_truth_sentence != decoded_preds[i]:
                wrong_cases.append((ground_truth_sentence, decoded_preds[i]))
                log_file.write("----------------------\n")
                log_file.write(f"Ground Truth: {ground_truth_sentence}\n")
                log_file.write(f"Model Prediction: {decoded_preds[i]}\n")

avg_val_loss = torch.mean(torch.tensor(val_loss_list))
test_wer = test_metric.wer
test_ser = test_metric.ser

log_file.write(f"Total CTC loss: {round(avg_val_loss.item(), 2)}, SER: {round(100*test_ser, 2)}, WER: {round(100*test_wer, 2)}")
log_file.close()
