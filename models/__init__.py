from .gru import GRUClassifier
from .resnet1d import ResNet1D
from .transformer import TransformerClassifier


def build_model(name, **kwargs):
    if name == "resnet":
        return ResNet1D(**kwargs)
    if name == "gru":
        return GRUClassifier(**kwargs)
    if name == "transformer":
        return TransformerClassifier(**kwargs)
    raise ValueError(f"Nieznany model: {name}")
