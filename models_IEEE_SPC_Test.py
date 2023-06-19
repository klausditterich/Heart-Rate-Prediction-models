import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np


def create_fourier_weights(signal_length):
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    return np.hstack([np.cos(theta_vals), -np.sin(theta_vals)])


def create_fourier_weightsconj(signal_length):
    k_vals, n_vals = np.mgrid[0:signal_length, 0:signal_length]
    theta_vals = 2 * np.pi * k_vals * n_vals / signal_length
    return np.hstack([np.cos(theta_vals), (-1)*(-np.sin(theta_vals))])


class Fourier(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(Fourier, self).__init__()

        self.filters = nn.Parameter(torch.tensor(np.hamming(1000), dtype=torch.float32).unsqueeze(0))

        self.fc = nn.Linear(hidden_size, 2000, bias=False)

        self.fc2 = nn.Linear(hidden_size, 2000, bias=False)

        self.fc3 = nn.Linear(500, 141)

        self.fc.weight.data = torch.tensor((create_fourier_weights(hidden_size)).transpose(1, 0),
                                           dtype=torch.float32).to(device)

        self.fc2.weight.data = torch.tensor((create_fourier_weightsconj(hidden_size)).transpose(1, 0),
                                            dtype=torch.float32).to(device)

        self.fc3.weight.data = torch.randn(500, 141).transpose(1, 0)

        self.fc.weight.requires_grad = False

        self.fc2.weight.requires_grad = False

        self.fc3.weight.requires_grad = True
        self.fc3.bias.requires_grad = True

        self.filters.requires_grad = False

        self.filters.data = torch.load("C:/Users/Usuario/Downloads/hamming_weights_ieeetest.pth")

    def forward(self, x):
        ham = x * self.filters
        out = self.fc(ham)
        out2 = self.fc2(ham)
        real_part = out[:, :1000]
        imag_part = out[:, 1000:]
        real_part2 = out2[:, :1000]
        imag_part2 = out2[:, 1000:]
        fft = torch.complex(real_part, imag_part)
        fftconj = torch.complex(real_part2, imag_part2)
        psd = torch.abs(fft * fftconj) / (1000 ** 2)

        freq_probs = self.fc3(psd[:, :500])

        return freq_probs


class CORNET(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CORNET, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.conv1 = nn.Conv1d(input_size, 32, 40)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm1d(32)
        self.max1 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(32, 32, 40)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm1d(32)
        self.max2 = nn.MaxPool1d(4)
        self.drop2 = nn.Dropout(0.1)
        self.lstm1 = nn.LSTM(50, 128, 2, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.fc = nn.Linear(4096, 1)

    def zero_hidden(self, device):
        self.h0_1 = torch.zeros(2, 20, 128).to(device)
        self.c0_1 = torch.zeros(2, 20, 128).to(device)

    def forward(self, x):
        out = self.conv1(x.transpose(1, 2))
        out = self.relu1(out)
        out = self.batch1(out)
        out = self.max1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)
        out = self.max2(out)
        out = self.drop2(out)
        out, _ = self.lstm1(out, (self.h0_1, self.c0_1))
        out = self.tanh1(out)
        out = out.reshape(20, -1)
        out = self.fc(out)
        return out


class CORNET2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CORNET2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.conv1 = nn.Conv1d(input_size, 32, 40)
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm1d(32)
        self.max1 = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(32, 32, 40)
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm1d(32)
        self.max2 = nn.MaxPool1d(4)
        self.drop2 = nn.Dropout(0.2)
        self.tcn = TemporalConvNet(50, [128, 128, 128, 128, 128, 128], 3, 0.1, False)
        self.tanh1 = nn.Tanh()
        self.fc = nn.Linear(4096, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batch1(out)
        out = self.max1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.batch2(out)
        out = self.max2(out)
        out = self.drop2(out)
        out = out.transpose(1, 2)
        out = self.tcn(out)
        out = self.tanh1(out)
        out = out.reshape(20, -1)
        out = self.fc(out)
        return out


def conv1d_same_padding(input, weight, bias=None, stride=1, dilation=1, groups=1):
    input_length = input.size(2)
    filter_length = weight.size(2)
    out_length = (input_length + stride[0] - 1) // stride[0]
    padding_length = max(0, (out_length - 1) * stride[0] + (filter_length - 1) * dilation[0] + 1 - input_length)

    length_odd = (padding_length % 2 != 0)
    if length_odd:
        input = F.pad(input, [0, int(length_odd)])

    return F.conv1d(input, weight, bias, stride, padding=padding_length // 2, dilation=dilation, groups=groups)


class Conv1dTCN(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, causal=True):
        self.causal = causal
        if causal:
            # double the output and chomp it
            padding = (kernel_size - 1) * dilation
        else:
            # set padding for zero for non-causal to padd in forward
            padding = 0
        super(Conv1dTCN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        if self.causal:
            x = F.conv1d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            x = x[:, :, :-self.padding[0]].contiguous()
            return x
        else:
            return conv1d_same_padding(input, self.weight, self.bias, self.stride, self.dilation, self.groups)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2, causal=True):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(Conv1dTCN(n_inputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation, causal=causal))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(Conv1dTCN(n_outputs, n_outputs, kernel_size,
                                           stride=stride, dilation=dilation, causal=causal))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, causal=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # which will add padding on both sides of the input (past and future)
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     dropout=dropout, causal=causal)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
