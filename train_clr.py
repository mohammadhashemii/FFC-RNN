import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from models.FFCRnn import FFCRnn
from torch.utils.data import DataLoader
from seqCLR.SeqCLR import SeqCLR
from seqCLR.InstanceMapper import InstanceMapper
from seqCLR.contrastive_learning_dataset import ContrastiveLearningDataset

# from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
# from models.resnet_simclr import ResNetSimCLR
# from simclr import SimCLR

parser = argparse.ArgumentParser()
# directories
parser.add_argument('--data_root', default='data', help='path to dataset directory')
parser.add_argument('--exp_dir', default='experiments', help='path to experiments directory')
# training
parser.add_argument('--exp', required=True, type=str, help='experiments number e.g. 01')
parser.add_argument('--resume', type=str, default=None, help='path to the checkpoint')
parser.add_argument('--pretrained', default=False, action="store_true", help='use pretrained model')
parser.add_argument('--finetune', default=False, action="store_true", help='fine-tune the model with lr=1e-5')
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
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')
parser.add_argument('--n_views', type=int, default=2, help='number of views')
parser.add_argument('--dataset_name', type=str, default='SadriDataset', help='path to the checkpoint')

args = parser.parse_args()


def main():
    # args = parser.parse_args()
    # assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # # check if gpu training is available
    # if not args.disable_cuda and torch.cuda.is_available():
    #     args.device = torch.device('cuda')
    #     cudnn.deterministic = True
    #     cudnn.benchmark = True
    # else:
    #     args.device = torch.device('cpu')
    #     args.gpu_index = -1

    dataset = ContrastiveLearningDataset(data_dir="data", img_size=(256, 32))

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views)
    # train_dataset = dataset.get_dataset('SadriDataset', 2)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    model = FFCRnn(nh=256,
                   output_number=42,
                   n_rnn=args.n_rnn,
                   feature_extractor=args.feature_extractor,
                   use_attention=args.use_attention)

    mapper = InstanceMapper(output_size=8, sequence_length=32, mode='adaptive', avg_dim=1, reshape=True)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(0):
        simclr = SeqCLR(model=model, optimizer=optimizer, scheduler=scheduler, mapper=mapper, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()
