# Deep Learning w Biometrii - VR Biometric Authentication

Projekt na spotkania #1, #2, #3 (Aleksander Sawicki).

Korpus: [VR-Biometric-Authentication](https://github.com/weibo053/VR-Biometric-Authentication/tree/4bbc555e6183327747cc29fd4463a71ab7894e09)
(commit `4bbc555`). Dane ruchu VR: 41 osob, 3 urzadzenia (Quest, Vive, Cosmos), 2 dni, 10 rzutow na sesje.
Shape pliku: `(41, 10, 135, 21)`.

## Juz zrobione

- `helpers.py` - ladowanie sesji, wybór modalnosci (pozycja / rotacja / full), normalizacja
- `datamodule.py` - PyTorch Lightning `VRBiometricDataModule`
  - train = dzien 1, test = dzien 2 (cross-day)
  - `modality`: `"position"`, `"rotation"`, `"full"`
  - zwraca `(x, y)` lub `(x, y, device_id)`

Przyklad:

```python
from datamodule import VRBiometricDataModule

dm = VRBiometricDataModule(modality="full", batch_size=32)
dm.setup()
batch = next(iter(dm.train_dataloader()))
```

## Podzial zadan

### Dominik Zabielski

1. **EDA**
   - charakterystyka zbioru, liczba probek, rozklad klas (osoby / urzadzenia / dni)
   - wykresy przykladowych trajektorii (pozycja kontrolerow i helmu)
   - podstawowe statystyki i obserwacje o jakosci danych
   - wynik: notatki + wykresy do raportu (folder `eda/`)

2. **Wybor modeli i trening**
   - wybierz i wytrenuj **trzy** klasyfikatory spośród: AlexNet-type CNN, Multi-Input Branch CNN, Transformer, GRU, LSTM, ResNet, InceptionTime
   - trening cross-day: dzien 1 train, dzien 2 test
   - zapis embeddingow: `train_embedding.npz`, `test_embedding.npz` (per model)
   - modele w `models/`, trening w `train_*.py`

### Hubert Zochowski

1. **Ewaluacja**
   - Rank-1 Identification
   - cosine similarity na embeddingach
   - skrypty metryk w `metrics.py`

2. **Eksperyment ablacyjny**
   - trening / ewaluacja na modalnosciach: `position`, `rotation`, `full`
   - porownanie Rank-1 / cosine dla kazdej modalnosci i modelu
   - DataModule juz to obsluguje: `VRBiometricDataModule(modality=...)`

3. **RSA**
   - Representational Similarity Analysis na embeddingach modeli
   - porownanie reprezentacji miedzy modelami
   - wnioski: ktore urzadzenie / modalnosc daje najwiecej informacji biometrycznej

## Wspolne ustalenia

- Uzywajcie `VRBiometricDataModule` z `datamodule.py`, nie piszcie wlasnego loadera.
- Modele trzymajcie w `models/`, trening w `train_*.py`, wyniki w `results/`.
- Embeddingi: `embeddings/<nazwa_modelu>/train_embedding.npz` oraz `test_embedding.npz`.
- Raport PDF: EDA + opis modeli + scenariusze + wyniki + wnioski.
- Kod + materialy do odtworzenia pakujecie w ZIP razem z raportem.

## Setup

```bash
pip install -r requirements.txt
```

Dane sa w `Dataset/MotionsNumpy/` (`Quest1/2.npy`, `Vive1/2.npy`, `Cosmos1/2.npy`).
`Prior1/2.npy` nie uzywamy w eksperymencie glownym.
