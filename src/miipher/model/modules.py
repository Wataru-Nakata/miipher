from torch import nn as nn
import torch
import math


class FiLMLayer(nn.Module):
    def __init__(self, input_channels, intermediate_channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_channels, intermediate_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            intermediate_channels, input_channels, kernel_size=3, stride=1, padding=1
        )
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        batch_size, K, D = a.size()
        Q = b.size(1)
        a = a.transpose(1, 2)
        output = self.conv2(
            (self.leaky_relu(self.conv1(a)).transpose(1, 2) + b).transpose(1, 2)
        )
        output = output.permute(0, 2, 1)
        assert output.size() == (batch_size, K, D)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[x]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Postnet(nn.Module):
    """Postnet
    - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(
        self,
        n_mel_channels,
        postnet_embedding_dim,
        postnet_kernel_size,
        postnet_n_convolutions,
    ):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = torch.nn.functional.dropout(
                torch.tanh(self.convolutions[i](x)), 0.5, self.training
            )
        x = torch.nn.functional.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.transpose(1, 2)
        return x
