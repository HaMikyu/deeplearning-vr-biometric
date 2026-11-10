from torch import nn


class GRUClassifier(nn.Module):
    def __init__(
        self,
        n_features=18,
        n_classes=41,
        embedding_dim=128,
        hidden_size=96,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.embedding = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        emb = self.embedding(pooled)
        logits = self.classifier(emb)
        return logits, emb
