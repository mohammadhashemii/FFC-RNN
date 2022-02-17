from torch.nn.functional import ctc_loss
import torch


def compute_ctc_loss(y_preds, y_train):
    # batch, T, C = y_preds.size()  # T = input length, C = vocab size + 2
    # y_preds = y_preds.permute(1, 0, 2)

    T, batch, C = y_preds.size()  # T = input length, C = vocab size + 2
    input_length = torch.IntTensor(batch).fill_(T)
    # target_length = torch.IntTensor([len(t) for t in y_train])
    target_length = torch.IntTensor([len(list(filter(lambda a: a != 0, t))) for t in y_train])

    loss = ctc_loss(y_preds, y_train, input_length, target_length, reduction='sum', zero_infinity=True)

    return loss
