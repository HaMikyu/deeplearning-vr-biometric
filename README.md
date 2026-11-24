# Deep Learning w Biometrii - VR Biometric Authentication

Projekt na spotkania #1, #2, #3 (Aleksander Sawicki).

Korpus: [VR-Biometric-Authentication](https://github.com/weibo053/VR-Biometric-Authentication/tree/4bbc555e6183327747cc29fd4463a71ab7894e09)
(commit `4bbc555`). Dane ruchu VR: 41 osob, 3 urzadzenia (Quest, Vive, Cosmos), 2 dni, 10 rzutow na sesje.
Shape pliku: `(41, 10, 135, 21)`.

> Wyniki w tabelach pochodza z kodu po poprawkach (rotacja 6D, Rank-k po osobach, EER,
> early stopping po `val_loss`, ablacja na 3 seedach). Instrukcja ponownego treningu: [TRENING.md](TRENING.md).

## Juz zrobione

- `helpers.py` - ladowanie sesji, wybór modalnosci (pozycja / rotacja / full), normalizacja
  - rotacja: katy Eulera -> macierz rotacji (kolejnosc `xyz`) -> reprezentacja 6D (2 kolumny macierzy),
    bo surowe katy zawijaja sie na +-pi i maja gimbal lock (patrz EDA, sekcja 10a)
  - liczba cech: `position` 9, `rotation` 18, `full` 27
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
  - trening cross-day, wczesne zatrzymanie i wybor modelu po `val_loss`
    (val Rank-1 z dnia 1 jest nasycony ~1.0 i nie rozroznia modeli)
  - zapis checkpointow, historii i metryk w `results/`
- `tune_models.py` - random search hiperparametrow, wybor po `val_loss`
- `metrics.py` - Rank-1 / Rank-5 (ranking po osobach), EER, cosine genuine / impostor, Rank-1 per urzadzenie
- `ablation.py` - model x modalnosc x seed (domyslnie 3 seedy), srednia i odchylenie
- `report_tables.py` - tabele markdown z `results/` do README i raportu
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

Zamiast grid search uzywamy **random search** z malym, stalym budzetem epok. Grid search rosnie wykladniczo z liczba parametrow i marnuje budzet na kombinacje, ktore i tak sa slabe. Random search lepiej pokrywa wazne wymiary, a w finalnym treningu dodatkowo dziala early stopping na `val_loss`.

```bash
python tune_models.py --model all --trials 6 --epochs 15
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

Na GPU caly zbior (ok. 1-2 MB) jest trzymany w pamieci karty i batchowany indeksami
(`TensorBatches` w `train_common.py`) zamiast przez `DataLoader`. Na RTX 3070 Ti: strojenie (3 modele x 6 prob x 15 epok)
ok. 2.5 min, trening trzech modeli ok. 50 s, ablacja 27 przebiegow ok. 5 min, RSA ok. 10 s
(na CPU ok. 17 s na epoke).

Przydatne flagi:

```bash
python train_resnet.py --epochs 80 --lr 0.0005
python train_resnet.py --no-test
```

## Wyniki

Wszystkie liczby ponizej pochodza z `results/` po ponownym treningu na poprawionym kodzie
(rotacja 6D, Rank-k po osobach, EER, wybor modelu po `val_loss`). Tabele generuje `python report_tables.py`.
Galeria = dzien 1 (train), probe = dzien 2 (test), podobienstwo cosinusowe embeddingow.

### Trening glowny (modalnosc full, seed 42)

| Model | Parametry | Best epoch | Val Rank-1 | Test Rank-1 | Test Rank-5 | Test EER |
|---|---:|---:|---:|---:|---:|---:|
| ResNet1D | 2,217,545 | 50 | 0.9959 | 0.9317 | 0.9813 | 0.0180 |
| GRUClassifier | 158,761 | 57 | 0.9715 | 0.7919 | 0.9488 | 0.0783 |
| TransformerClassifier | 373,769 | 38 | 1.0000 | 0.9569 | 0.9927 | 0.0195 |

Val jest z dnia 1 i sluzy tylko do wyboru modelu. Test z dnia 2 jest jedynym wynikiem cross-day.
Spadek val -> test jest nadal widoczny (najwiekszy dla GRU: 0.97 -> 0.79), ale mniejszy niz przed poprawkami.

## Ewaluacja, ablacja modalnosci i RSA

```bash
python ablation.py   # 3 modele x 3 modalnosci x 3 seedy -> results/ablation.csv, ablation_runs.csv, embeddings/<model>_<modalnosc>/
python rsa.py        # results/rsa_models.csv, rsa_reference.csv, rsa_models.png
```

Metryki sa w `metrics.py` (Rank-1, Rank-5, EER, cosine similarity). Hiperparametry strojone na `full`
uzyte dla wszystkich modalnosci. Seed zmienia inicjalizacje, kolejnosc batchy i podzial walidacji.
Kazdy przebieg osobno: `results/ablation_runs.csv`.

### Ablacja (srednia ± odchylenie z 3 seedow: 42, 43, 44; dzien 2)

| Model | Modalnosc | Rank-1 | Rank-5 | EER | cos genuine | cos impostor | Rank-1 Quest | Rank-1 Vive | Rank-1 Cosmos |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet1D | position | 0.840 ± 0.013 | 0.948 ± 0.009 | 0.050 ± 0.007 | 0.875 ± 0.004 | 0.472 ± 0.007 | 0.902 ± 0.017 | 0.829 ± 0.013 | 0.788 ± 0.024 |
| ResNet1D | rotation | 0.871 ± 0.015 | 0.968 ± 0.005 | 0.037 ± 0.005 | 0.887 ± 0.005 | 0.481 ± 0.004 | 0.924 ± 0.006 | 0.877 ± 0.029 | 0.813 ± 0.019 |
| ResNet1D | full | 0.927 ± 0.014 | 0.982 ± 0.005 | 0.021 ± 0.003 | 0.919 ± 0.003 | 0.468 ± 0.003 | 0.962 ± 0.012 | 0.933 ± 0.028 | 0.885 ± 0.016 |
| GRUClassifier | position | 0.619 ± 0.009 | 0.865 ± 0.006 | 0.161 ± 0.007 | 0.799 ± 0.003 | 0.418 ± 0.009 | 0.712 ± 0.022 | 0.628 ± 0.008 | 0.515 ± 0.018 |
| GRUClassifier | rotation | 0.738 ± 0.026 | 0.911 ± 0.014 | 0.121 ± 0.007 | 0.829 ± 0.007 | 0.421 ± 0.014 | 0.793 ± 0.061 | 0.737 ± 0.017 | 0.683 ± 0.032 |
| GRUClassifier | full | 0.815 ± 0.021 | 0.951 ± 0.006 | 0.076 ± 0.007 | 0.852 ± 0.003 | 0.396 ± 0.003 | 0.879 ± 0.008 | 0.824 ± 0.041 | 0.741 ± 0.030 |
| TransformerClassifier | position | 0.839 ± 0.024 | 0.953 ± 0.005 | 0.077 ± 0.010 | 0.816 ± 0.010 | 0.299 ± 0.002 | 0.880 ± 0.011 | 0.855 ± 0.037 | 0.780 ± 0.024 |
| TransformerClassifier | rotation | 0.901 ± 0.016 | 0.972 ± 0.010 | 0.045 ± 0.012 | 0.846 ± 0.012 | 0.289 ± 0.003 | 0.948 ± 0.017 | 0.929 ± 0.025 | 0.826 ± 0.013 |
| TransformerClassifier | full | 0.938 ± 0.017 | 0.991 ± 0.002 | 0.026 ± 0.006 | 0.859 ± 0.008 | 0.279 ± 0.002 | 0.965 ± 0.016 | 0.964 ± 0.023 | 0.884 ± 0.026 |

### RSA z tozsamoscia i urzadzeniem (seed 42, dzien 2)

Spearman miedzy RDM-ami `1 - cosine` na dniu 2; pelna macierz model-model w `results/rsa_models.csv`
i `results/rsa_models.png`.

| Przebieg | RSA z tozsamoscia | RSA z urzadzeniem |
|---|---:|---:|
| resnet_position | 0.249 | 0.019 |
| resnet_rotation | 0.254 | 0.006 |
| resnet_full | 0.259 | 0.004 |
| gru_position | 0.213 | 0.087 |
| gru_rotation | 0.225 | 0.029 |
| gru_full | 0.244 | 0.044 |
| transformer_position | 0.238 | 0.048 |
| transformer_rotation | 0.254 | 0.005 |
| transformer_full | 0.260 | 0.014 |

### Wnioski

- **Modalnosc:** `full` > `rotation` > `position` dla kazdego modelu (Rank-1, Rank-5, EER i RSA z tozsamoscia).
  **To odwrotnie niz przed poprawkami** (wtedy `position` > `rotation`): surowe katy Eulera z zawijaniem
  na ±pi i gimbal lockiem psuly sygnal rotacji. Po zamianie na reprezentacje 6D rotacja niesie wiecej
  informacji biometrycznej niz pozycja (np. GRU 0.738 vs 0.619, Transformer 0.901 vs 0.839),
  a obie modalnosci sa komplementarne (`full` najlepsza wszedzie).
- **Urzadzenie:** Cosmos ma najnizszy Rank-1 we wszystkich 9 konfiguracjach (model x modalnosc).
  Quest ma najwyzszy Rank-1 w 9 z 9 (dla Transformer full praktycznie remis z Vive: 0.965 vs 0.964).
- **Modele:** Transformer i ResNet1D sa porownywalne na `full` (Rank-1 0.938 ± 0.017 vs 0.927 ± 0.014,
  roznica w granicach odchylenia; EER 0.026 vs 0.021). Transformer ma ok. 6x mniej parametrow
  i najlepiej separuje osoby (najnizsze cos impostor ~0.28). GRU jest wyraznie najslabszy (0.815, EER 0.076).
- **RSA model-model:** reprezentacje sa podobne glownie w obrebie tej samej modalnosci
  (gru_position vs transformer_position 0.68, gru_rotation vs transformer_rotation 0.65),
  a pozycja i rotacja daja rozne geometrie (0.16-0.25). GRU i Transformer sa do siebie blizsze
  niz kazdy z nich do ResNet1D.
- **RSA z urzadzeniem:** pozycja koduje urzadzenie wyraznie mocniej niz rotacja we wszystkich modelach
  (np. GRU 0.087 vs 0.029, Transformer 0.048 vs 0.005) - pozycje zaleza od ukladu wspolrzednych
  trackingu danego zestawu, orientacja mniej. GRU najmocniej koduje urzadzenie i ma najnizszy Rank-1.

## Wyniki przed poprawkami (archiwalne)

Ponizsze tabele sa ze starego kodu: surowe katy Eulera, Rank-5 liczony po probkach (zanizony),
wybor modelu po val Rank-1, jeden seed. Zostawione do porownania w raporcie.

Klasyfikacja cross-day, galeria z dnia 1, probe z dnia 2:

| Model | Val Rank-1 | Test Rank-1 | Test Rank-5 |
|---|---:|---:|---:|
| ResNet1D | 1.0000 | 0.8862 | 0.8976 |
| GRUClassifier | 0.9756 | 0.6780 | 0.7463 |
| TransformerClassifier | 0.9959 | 0.8569 | 0.9122 |

Ablacja (jeden seed, surowe katy Eulera):

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

RSA (stare):

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

Zmiana wzgledem starych wynikow: Rank-1 na `full` wzrosl dla wszystkich modeli
(ResNet 0.900 -> 0.927, GRU 0.663 -> 0.815, Transformer 0.922 -> 0.938, srednie z 3 seedow),
a najwiekszy zysk dala rotacja (GRU 0.522 -> 0.738, Transformer 0.767 -> 0.901).

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

Z karta NVIDIA najpierw torch z CUDA (inaczej pip zostawi wersje CPU), np.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Dane sa w `Dataset/MotionsNumpy/` (`Quest1/2.npy`, `Vive1/2.npy`, `Cosmos1/2.npy`).
`Prior1/2.npy` nie uzywamy w eksperymencie glownym.
