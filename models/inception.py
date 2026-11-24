"""InceptionTime (Ismail Fawaz i in. 2020) dla sekwencji (B, T, C).

Kazdy modul: bottleneck 1x1 -> trzy rownolegle konwolucje o roznych
dlugosciach jader + galaz max-pool -> konkatenacja. Co 3 moduly polaczenie
rezydualne. Jadra skrocone do 39/19/9 (sekwencja ma 135 klatek).
"""

import torch
from torch import nn

from .heads import embedding_layers


class InceptionModule(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernels=(39, 19, 9), bottleneck=32):
        super().__init__()
        use_bottleneck = in_channels > 1
        self.bottleneck = (
            nn.Conv1d(in_channels, bottleneck, 1, bias=False)
            if use_bottleneck
            else nn.Identity()
        )
        mid = bottleneck if use_bottleneck else in_channels
        self.convs = nn.ModuleList(
            nn.Conv1d(mid, n_filters, k, padding=k // 2, bias=False) for k in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, 1, bias=False),
        )
        out = n_filters * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.bottleneck(x)
        out = [conv(z) for conv in self.convs] + [self.pool_branch(x)]
        return self.act(self.bn(torch.cat(out, dim=1)))


class InceptionTime(nn.Module):
    def __init__(
        self,
        n_features=27,
        n_classes=41,
        embedding_dim=128,
        n_filters=32,
        depth=6,
        dropout=0.1,
        linear_embedding=False,
    ):
        super().__init__()
        channels = n_filters * 4
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        in_ch = n_features
        res_in = n_features
        for i in range(depth):
            self.blocks.append(InceptionModule(in_ch, n_filters))
            in_ch = channels
            if i % 3 == 2:
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv1d(res_in, channels, 1, bias=False),
                        nn.BatchNorm1d(channels),
                    )
                )
                res_in = channels
        self.act = nn.ReLU(inplace=True)
        self.embedding = nn.Sequential(
            *embedding_layers(channels, embedding_dim, dropout, linear_embedding)
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        res = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                x = self.act(x + self.shortcuts[i // 3](res))
                res = x
        emb = self.embedding(x.mean(dim=2))
        return self.classifier(emb), emb
