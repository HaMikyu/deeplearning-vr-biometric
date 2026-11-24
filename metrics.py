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


def identity_scores(similarity, gallery_labels):
    """Podobienstwo do kazdej osoby = max po jej probkach w galerii.

    Zwraca (scores (n_query, n_ids), ids).
    """
    ids = np.unique(gallery_labels)
    scores = np.stack(
        [similarity[:, gallery_labels == i].max(axis=1) for i in ids], axis=1
    )
    return scores, ids


def rank_k(similarity, query_labels, gallery_labels, k=1):
    """Rank-k Identification: czy wlasciwa osoba jest wsrod k najblizszych OSOB.

    Ranking jest po tozsamosciach, a nie po probkach - inaczej k najblizszych
    probek moze nalezec do jednej osoby i Rank-5 bylby prawie rowny Rank-1.
    """
    scores, ids = identity_scores(similarity, gallery_labels)
    top = ids[np.argsort(-scores, axis=1)[:, :k]]
    return float((top == query_labels[:, None]).any(axis=1).mean())


def rank1(similarity, query_labels, gallery_labels):
    """Rank-1 Identification: osoba najblizszej probki z galerii."""
    return rank_k(similarity, query_labels, gallery_labels, k=1)


def cosine_genuine_impostor(similarity, query_labels, gallery_labels):
    """Srednie cosine dla par tej samej osoby i par roznych osob."""
    same = query_labels[:, None] == gallery_labels[None, :]
    return float(similarity[same].mean()), float(similarity[~same].mean())


def eer(similarity, query_labels, gallery_labels):
    """Equal Error Rate weryfikacji na parach (probe, probka z galerii).

    Prog na podobienstwie cosinusowym: FAR = odsetek par roznych osob
    powyzej progu, FRR = odsetek par tej samej osoby ponizej progu.
    """
    same = (query_labels[:, None] == gallery_labels[None, :]).ravel()
    scores = similarity.ravel()
    order = np.argsort(-scores, kind="stable")
    same = same[order]
    n_genuine = same.sum()
    n_impostor = len(same) - n_genuine
    # prog przesuwany od najwyzszego podobienstwa w dol
    tpr = np.cumsum(same) / n_genuine
    far = np.cumsum(~same) / n_impostor
    frr = 1.0 - tpr
    i = np.argmin(np.abs(far - frr))
    return float((far[i] + frr[i]) / 2)


def ranking_metrics(query, query_labels, gallery, gallery_labels):
    similarity = cosine_similarity(query, gallery)
    return {
        "rank1": rank_k(similarity, query_labels, gallery_labels, 1),
        "rank5": rank_k(similarity, query_labels, gallery_labels, 5),
    }


def evaluate(query, query_labels, query_devices, gallery, gallery_labels):
    similarity = cosine_similarity(query, gallery)
    genuine, impostor = cosine_genuine_impostor(similarity, query_labels, gallery_labels)
    out = {
        "rank1": rank_k(similarity, query_labels, gallery_labels, 1),
        "rank5": rank_k(similarity, query_labels, gallery_labels, 5),
        "eer": eer(similarity, query_labels, gallery_labels),
        "cos_genuine": genuine,
        "cos_impostor": impostor,
    }
    for device_id, name in enumerate(DEVICES):
        mask = query_devices == device_id
        out[f"rank1_{name.lower()}"] = rank1(
            similarity[mask], query_labels[mask], gallery_labels
        )
    return out
