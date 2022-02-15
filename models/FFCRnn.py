import torch.nn as nn
from FFC import *
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, out_features, return_seq=True):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_features)
        self.return_seq = return_seq

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)

        if self.return_seq:
            output = self.embedding(t_rec)  # [t * b, nOut]
            output = output.view(t, b, -1)
        else:
            output = t_rec.view(t, b, -1)

        return output


class FFCRnn(nn.Module):

    def __init__(self, image_height, nc, output_number, nh, n_rnn=2, leaky_relu=False):
        super(FFCRnn, self).__init__()

        assert image_height % 16 == 0, 'imgH has to be a multiple of 16'

        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        padding_sizes = [1, 1, 1, 1, 1, 1, 0]
        stride_sizes = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]
        nm = [32, 32, 48, 64, 80, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            input_channels = nc if i == 0 else nm[i - 1]
            output_channels = nm[i]

            # TODO REPLACE CONV WITH FFC
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(input_channels, output_channels, (kernel_sizes[i], kernel_sizes[i]),
                                     (stride_sizes[i], stride_sizes[i]), padding_sizes[i]))

            # cnn.add_module('conv{0}'.format(i),
            #                nn.Conv2d(input_channels, output_channels, (kernel_sizes[i], kernel_sizes[i]),
            #                          (stride_sizes[i], stride_sizes[i]), padding_sizes[i]))

            if batch_normalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(output_channels))
            if leaky_relu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        conv_relu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64 x 16 x64
        conv_relu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128 x 8 x 32
        conv_relu(2)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 1)))  # 128 x 8 x 32
        conv_relu(3)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((1, 2)))  # 256 x 4 x 16
        conv_relu(4)
        # cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((1, 2)))

        # conv_relu(5)
        # cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512 x 2 x 16
        # conv_relu(6, True)  # 512 x 1 x 16

        self.cnn = cnn

        self.adp = nn.AdaptiveAvgPool2d((512, None))

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, output_number, return_seq=False))

        self.fc = nn.Linear(nh * 2, output_number)

    def forward(self, x):
        # conv features
        conv = self.cnn(x)

        b, c, h, w = conv.size()
        # print(b, c, h, w)
        # assert h == 1, "the height of conv must be 1"
        print(conv.size())
        # conv = conv.squeeze(2)
        conv = conv.view(b, c * h, w)
        print(conv.size())

        conv = self.adp(conv)

        conv = conv.permute(2, 0, 1)  # [w, b, c]

        print(conv.size())
        # rnn features
        output = self.rnn(conv)
        print(conv.size())
        output = torch.stack([F.log_softmax(self.fc(output[i]), dim=-1) for i in range(output.shape[0])])

        # in order to be compatible with our ctc
        output = output.permute(1, 0, 2)  # (batch, ctc input length, output number)
        print(output.size())
        return output


if __name__ == '__main__':
    # ffc_rnn = FFCRnn(32, 32, 32, 32)
    ffc_rnn = FFCRnn(32, 1, 41, 256)
    tensor = torch.zeros([10, 1, 32, 256], dtype=torch.float32)
    res = ffc_rnn(tensor)
