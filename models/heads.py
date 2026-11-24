from torch import nn


def embedding_layers(in_dim, embedding_dim, dropout, linear=False):
    """Warstwy glowicy embeddingu.

    linear=False (domyslnie, baseline): Linear -> ReLU -> Dropout.
    linear=True (SOTA / ArcFace): Linear -> BatchNorm1d. Bez ReLU, bo
    nieujemny embedding marnuje polowe sfery i podbija cos impostor.
    """
    if linear:
        return [nn.Linear(in_dim, embedding_dim), nn.BatchNorm1d(embedding_dim)]
    return [
        nn.Linear(in_dim, embedding_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
    ]
