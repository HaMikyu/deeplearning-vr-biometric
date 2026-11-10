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
  - walidacja stratyfikowana po `(osoba, urzadzenie)`: `val_per_group=2` rzuty na grupe
    (domyslnie 984 train / 246 val), wiec kazda osoba i urzadzenie sa w val
  - `val` sluzy tylko do early stopingu / wyboru hiperparametrow, metryki raportujemy na dniu 2
  - zwraca `(x, y)` lub `(x, y, device_id)`
- `EDA/EDA.ipynb` - pelna EDA z wykresami w `EDA/figures/`
- `models/` - trzy klasyfikatory: `ResNet1D`, `GRUClassifier`, `TransformerClassifier`
- `train_common.py` + `train_resnet.py`, `train_gru.py`, `train_transformer.py`
  - trening cross-day, wczesne zatrzymanie na val Rank-1
  - zapis checkpointow, historii i metryk w `results/`
- `tune_models.py` - random search hiperparametrow z wczesnym zatrzymaniem
- `embeddings/<model>/` - `train_embedding.npz`, `val_embedding.npz`, `test_embedding.npz`

Przyklad:

```python
from datamodule import VRBiometricDataModule

dm = VRBiometricDataModule(modality="full", batch_size=32)
dm.setup()
batch = next(iter(dm.train_dataloader()))
```

## Wybrane modele

| Model | Kategoria z listy | Plik | Glowna rola |
|---|---|---|---|
| ResNet1D | ResNet | `models/resnet1d.py` | lokalne wzorce ruchu i stabilny trening dzieki polaczeniom rezydualnym |
| GRUClassifier | GRU | `models/gru.py` | dynamika czasowa i zaleznosci sekwencyjne przy malej liczbie parametrow |
| TransformerClassifier | Transformer | `models/transformer.py` | zaleznosci dlugiego zasiegu i globalny kontekst calej trajektorii |

Uzasadnienie wyboru:

- sa to **trzy rozne kategorie** z listy, wiec porownanie nie jest zdominowane przez jeden typ indukcyjnego biasu,
- **ResNet1D** dobrze wychwytuje lokalne wzorce (przyspieszenia, zmiany kierunku) i jest odporny na glebokosc dzieki skip-connections,
- **GRU** modeluje kolejnosc zdarzen i jest lzejszy od LSTM przy podobnej sile na krotkich sekwencjach,
- **Transformer** dodaje mechanizm self-attention, ktory laczy odlegle fragmenty rzutu i daje najbardziej odmienne embeddingi do RSA,
- taki zestaw daje sensowne porownanie reprezentacji: konwolucyjna, rekurencyjna i attention-based.

## Strojenie hiperparametrow

Zamiast grid search uzywamy **random search** z malym, stalym budzetem epok. Grid search rosnie wykladniczo z liczba parametrow i marnuje budzet na kombinacje, ktore i tak sa slabe. Random search lepiej pokrywa wazne wymiary, a w finalnym treningu dodatkowo dziala early stopping na val Rank-1.

```bash
python tune_models.py --model all --trials 4 --epochs 15
```

Wyniki trafiaja do `results/tuning_<model>.csv`, a najlepsza konfiguracja do `results/best_<model>.json`. Skrypty treningowe automatycznie ja wczytuja przez `configs.get_config`.

Alternatywy, ktore mozna dodac pozniej: successive halving / Hyperband oraz Optuna TPE. Random search jest tu wystarczajacy, bo przestrzen jest mala, a trening jednego modelu jest tani.

## Trening

```bash
python train_resnet.py
python train_gru.py
python train_transformer.py
```

Kazdy skrypt:

- trenuje na dniu 1 i waliduje na stratyfikowanym podziale dnia 1,
- raportuje koncowy wynik na dniu 2,
- zapisuje checkpoint do `models/checkpoints/`,
- zapisuje metryki i historie do `results/`,
- zapisuje embeddingi do `embeddings/<model>/`.

Przydatne flagi:

```bash
python train_resnet.py --epochs 80 --lr 0.0005
python train_resnet.py --no-test
```

## Wyniki

Klasyfikacja cross-day, galeria z dnia 1, probe z dnia 2:

| Model | Val Rank-1 | Test Rank-1 | Test Rank-5 |
|---|---:|---:|---:|
| ResNet1D | 1.0000 | 0.8862 | 0.8976 |
| GRUClassifier | 0.9756 | 0.6780 | 0.7463 |
| TransformerClassifier | 0.9959 | 0.8569 | 0.9122 |

Val jest z dnia 1 i sluzy tylko do wyboru modelu. Test z dnia 2 jest jedynym wynikiem cross-day. Widac wyrazny spadek miedzy val i test, szczegolnie dla GRU, co jest oczekiwane przy zmianie dnia i pokazuje, ze sama walidacja w obrebie dnia 1 nie wystarcza do oceny biometrycznej.

## Podzial zadan

### Dominik Zabielski

1. **EDA**
   - charakterystyka zbioru, liczba probek, rozklad klas (osoby / urzadzenia / dni)
   - wykresy przykladowych trajektorii (pozycja kontrolerow i helmu)
   - podstawowe statystyki i obserwacje o jakosci danych
   - wynik: notatki + wykresy do raportu (folder `EDA/`)

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
  Dodatkowo zapisujemy `val_embedding.npz`.
- Raport PDF: EDA + opis modeli + scenariusze + wyniki + wnioski.
- Kod + materialy do odtworzenia pakujecie w ZIP razem z raportem.

## Setup

```bash
pip install -r requirements.txt
```

Dane sa w `Dataset/MotionsNumpy/` (`Quest1/2.npy`, `Vive1/2.npy`, `Cosmos1/2.npy`).
`Prior1/2.npy` nie uzywamy w eksperymencie glownym.
