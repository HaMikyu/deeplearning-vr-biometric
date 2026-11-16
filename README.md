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

## Ewaluacja, ablacja modalnosci i RSA

```bash
python ablation.py   # 3 modele x 3 modalnosci -> results/ablation.csv, embeddings/<model>_<modalnosc>/
python rsa.py        # results/rsa_models.csv, rsa_reference.csv, rsa_models.png
```

Metryki sa w `metrics.py` (Rank-1, cosine similarity). Galeria = dzien 1, probe = dzien 2,
jeden seed (42), hiperparametry strojone na `full` uzyte dla wszystkich modalnosci.
Wyniki `full` roznia sie nieco od tabeli wyzej, bo to osobny przebieg treningu.

| Model | Modalnosc | Rank-1 | cos genuine | cos impostor | Rank-1 Quest | Rank-1 Vive | Rank-1 Cosmos |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet1D | position | 0.846 | 0.880 | 0.445 | 0.907 | 0.841 | 0.788 |
| ResNet1D | rotation | 0.823 | 0.853 | 0.449 | 0.910 | 0.827 | 0.732 |
| ResNet1D | full | 0.900 | 0.912 | 0.449 | 0.922 | 0.917 | 0.861 |
| GRU | position | 0.619 | 0.799 | 0.413 | 0.744 | 0.595 | 0.517 |
| GRU | rotation | 0.522 | 0.751 | 0.426 | 0.541 | 0.598 | 0.427 |
| GRU | full | 0.663 | 0.819 | 0.445 | 0.717 | 0.707 | 0.563 |
| Transformer | position | 0.888 | 0.847 | 0.337 | 0.915 | 0.941 | 0.807 |
| Transformer | rotation | 0.767 | 0.780 | 0.372 | 0.790 | 0.810 | 0.702 |
| Transformer | full | 0.922 | 0.859 | 0.354 | 0.956 | 0.946 | 0.863 |

RSA (Spearman miedzy RDM-ami `1 - cosine` na dniu 2; pelna macierz w `results/rsa_models.csv`):

| Przebieg | RSA z tozsamoscia | RSA z urzadzeniem |
|---|---:|---:|
| resnet_position | 0.248 | 0.018 |
| resnet_rotation | 0.244 | 0.011 |
| resnet_full | 0.257 | 0.009 |
| gru_position | 0.211 | 0.085 |
| gru_rotation | 0.184 | 0.047 |
| gru_full | 0.227 | 0.058 |
| transformer_position | 0.255 | 0.020 |
| transformer_rotation | 0.233 | 0.014 |
| transformer_full | 0.258 | 0.010 |

Wnioski:

- **Modalnosc:** `full` jest najlepsza dla kazdego modelu, a sama pozycja jest lepsza niz sama rotacja
  (Rank-1 i RSA z tozsamoscia). Rotacja wnosi informacje uzupelniajaca: `full` > `position`.
- **Urzadzenie:** Cosmos wypada najgorzej we wszystkich 9 przebiegach. Quest i Vive sa zblizone,
  Quest jest najlepszy w 6 z 9 przebiegow.
- **Modele:** reprezentacje sa podobne glownie w obrebie tej samej modalnosci (np. gru_position vs
  transformer_position 0.51), a pozycja i rotacja daja wyraznie rozne geometrie (0.11-0.22).
  GRU i Transformer sa do siebie blizsze niz kazdy z nich do ResNet1D.
- GRU najmocniej koduje urzadzenie (RSA 0.05-0.09 wobec ok. 0.01-0.02) i ma najnizszy Rank-1.

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
