"""Metryki na embeddingach: galeria = dzien 1 (train), probe = dzien 2 (test)."""

import numpy as np

from helpers import DEVICES


def load_embedding(path):
    data = np.load(path)
    return data["embeddings"], data["labels"], data["device_ids"]


def cosine_similarity(query, gallery):
    """Macierz podobienstwa cosinusowego (n_query, n_gallery)."""
    query = query / np.linalg.norm(query, axis=1, keepdims=True).clip(1e-12)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True).clip(1e-12)
    return query @ gallery.T


def rank1(similarity, query_labels, gallery_labels):
    """Rank-1 Identification: osoba najblizszej probki z galerii."""
    predicted = gallery_labels[similarity.argmax(axis=1)]
    return float((predicted == query_labels).mean())


def cosine_genuine_impostor(similarity, query_labels, gallery_labels):
    """Srednie cosine dla par tej samej osoby i par roznych osob."""
    same = query_labels[:, None] == gallery_labels[None, :]
    return float(similarity[same].mean()), float(similarity[~same].mean())


def evaluate(query, query_labels, query_devices, gallery, gallery_labels):
    similarity = cosine_similarity(query, gallery)
    genuine, impostor = cosine_genuine_impostor(similarity, query_labels, gallery_labels)
    out = {
        "rank1": rank1(similarity, query_labels, gallery_labels),
        "cos_genuine": genuine,
        "cos_impostor": impostor,
    }
    for device_id, name in enumerate(DEVICES):
        mask = query_devices == device_id
        out[f"rank1_{name.lower()}"] = rank1(
            similarity[mask], query_labels[mask], gallery_labels
        )
    return out
