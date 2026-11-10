from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class ResNet1D(nn.Module):
    def __init__(
        self,
        n_features=18,
        n_classes=41,
        embedding_dim=128,
        base_channels=64,
        dropout=0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                n_features,
                base_channels,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(
            base_channels,
            base_channels,
            blocks=2,
            stride=1,
            dropout=dropout,
        )
        self.layer2 = self._make_layer(
            base_channels,
            base_channels * 2,
            blocks=2,
            stride=2,
            dropout=dropout,
        )
        self.layer3 = self._make_layer(
            base_channels * 2,
            base_channels * 4,
            blocks=2,
            stride=2,
            dropout=dropout,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dropout):
        layers = [ResidualBlock(in_channels, out_channels, stride, dropout)]
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, 1, dropout)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        emb = self.embedding(x)
        logits = self.classifier(emb)
        return logits, emb
