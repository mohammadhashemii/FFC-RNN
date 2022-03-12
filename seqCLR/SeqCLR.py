import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SeqCLR(object):

    def __init__(self, mapper, temperature=0.07, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.device)
        self.temperature = temperature
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.mapper = mapper
        # self.writer = SummaryWriter()
        # logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
        # labels = torch.FloatTensor(labels)
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        # save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        # logging.info(f"Start SimCLR training for {self.args.n_epochs} epochs.")
        # logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.n_epochs):
            total_loss = 0.0
            total_acc_top1 = 0
            total_acc_top5 = 0
            n_total_samples = 0

            for images in tqdm(train_loader):

                # print(images.size())

                images = torch.cat(images, dim=0)
                n_total_samples = len(train_loader) * len(images)

                images = images.to(self.device, dtype=torch.float32)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    features = self.mapper(features)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                n_iter += 1

                total_loss += loss.item()
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                total_acc_top1 += top1*len(logits)
                total_acc_top5 += top5*len(logits)


            print(f"[Epoch{epoch_counter}/{self.args.n_epochs}]: Loss: {total_loss/len(train_loader)},"
                  f" Accuracy Top1: {total_acc_top1/n_total_samples}, Accuracy Top5: {total_acc_top5/n_total_samples}")
                # if n_iter % self.args.log_every_n_steps == 0:
                # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                # self.writer.add_scalar('loss', loss, global_step=n_iter)
                # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                # self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        # logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_path = os.path.join("checkpoint_last.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        # save_checkpoint({
        #     'epoch': self.args.epochs,
        #     'arch': self.args.arch,
        #     'state_dict': self.model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        # }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        # logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
