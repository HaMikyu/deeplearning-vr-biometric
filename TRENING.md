# Instrukcja: ponowny trening po poprawkach

Ten plik jest dla osoby (albo agenta Claude Code), która odpala trening na
domowym komputerze. Wyniki w `results/` i tabele w `README.md` pochodzą jeszcze
ze **starego kodu**, więc trzeba wszystko przeliczyć.

## Co się zmieniło w kodzie (dlaczego trzeba trenować od nowa)

1. **Rotacja jako macierz 6D zamiast kątów Eulera** (`helpers.py: encode_rotation`).
   Surowe kąty zawijały się na ±π i miały gimbal lock: skok > 1 rad między klatkami
   w 90% nagrań prawego kontrolera, 29% hełmu i 13% lewego. Teraz kąty są zamieniane
   na macierz rotacji (kolejność `xyz`), z której bierzemy 2 kolumny.
   Liczba cech: `position` 9, `rotation` 18, `full` 27 (wcześniej 9 / 9 / 18).
2. **Rank-k po tożsamościach** (`metrics.py: rank_k`). Stary Rank-5 brał 5 najbliższych
   *próbek*, które często należały do jednej osoby, więc Rank-5 ≈ Rank-1.
3. **Nowa metryka weryfikacji: EER** (`metrics.py: eer`), liczona na podobieństwie cosinusowym.
4. **Wczesne zatrzymanie i strojenie po `val_loss`** zamiast val Rank-1, który na dniu 1
   od razu dochodzi do ~1.0 i nie rozróżnia modeli (`train_common.py`, `tune_models.py`).
5. **Ablacja na kilku seedach** (`ablation.py --seeds`, domyślnie 42 43 44): zapisuje
   `results/ablation_runs.csv` (każdy przebieg) i `results/ablation.csv` (średnia i odchylenie).
6. **`report_tables.py`** drukuje gotowe tabele markdown do README / raportu.
7. EDA: nowa sekcja 10a o ciągłości rotacji i wykres `EDA/figures/rotacja_euler_vs_6d.png`.

## Krok po kroku

Wszystkie komendy uruchamiaj z katalogu głównego repo.

### 1. Pobranie kodu

```bash
git clone https://github.com/HaMikyu/deeplearning-vr-biometric
cd deeplearning-vr-biometric
git checkout fix/rotation-metrics
```

(Jeśli repo już jest: `git fetch && git checkout fix/rotation-metrics && git pull`.)

### 2. Środowisko

Python 3.10+. Jeśli jest karta NVIDIA, najpierw zainstaluj torch z CUDA
(sprawdź komendę na https://pytorch.org/get-started/locally/), np.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Potem reszta zależności:

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

### 3. Szybki test (ok. 1–2 min)

```bash
python datamodule.py
```

Oczekiwane: `features: 27`, `sizes: 984 246 1230`, `val groups: 123 expected: 123`.

### 4. Strojenie hiperparametrów

Trzeba stroić od nowa, bo zmieniła się liczba cech i kryterium wyboru.

```bash
python tune_models.py --model all --trials 6 --epochs 15
```

Nadpisuje `results/tuning_*.csv` i `results/best_*.json`.

### 5. Trening trzech modeli (modalność `full`)

```bash
python train_resnet.py --no-dashboard
python train_gru.py --no-dashboard
python train_transformer.py --no-dashboard
```

Zapisuje `results/<model>_metrics.json`, `results/<model>_history.csv`,
`embeddings/<model>/{train,val,test}_embedding.npz` i `models/checkpoints/<model>.pt`.

### 6. Ablacja (3 modele × 3 modalności × 3 seedy = 27 treningów)

```bash
python ablation.py
```

Najdłuższy krok. Wyniki są zapisywane po każdym przebiegu do `results/ablation_runs.csv`.

### 7. RSA

```bash
python rsa.py
```

Zapisuje `results/rsa_models.csv`, `results/rsa_reference.csv`, `results/rsa_models.png`.

### 8. EDA (opcjonalnie, żeby notebook miał wyniki nowej sekcji 10a)

```bash
cd EDA
jupyter nbconvert --to notebook --execute --inplace EDA.ipynb
cd ..
```

### 9. Tabele do README

```bash
python report_tables.py
```

Wklej wynik do `README.md` w miejsce tabel w sekcjach **Wyniki** oraz
**Ewaluacja, ablacja modalnosci i RSA**. Następnie przepisz **Wnioski** na podstawie
nowych liczb. Sprawdź zwłaszcza, czy nadal zachodzą:
- `full` > `position` > `rotation` (po poprawce rotacji może się to zmienić),
- Cosmos najgorszy, Quest/Vive najlepsze,
- GRU najsłabszy i najmocniej kodujący urządzenie (RSA z urządzeniem).

Opisz w raporcie zmianę względem starych wyników (stare tabele są w sekcji
„Wyniki przed poprawkami” w README).

### 10. Commit i push

```bash
git add -A
git commit -m "Wyniki po poprawkach: rotacja 6D, Rank-k po osobach, EER, 3 seedy"
git push
```

`embeddings/` i `*.pt` są w `.gitignore`. Do ZIP-a z raportem dołącz je ręcznie
(zadanie wymaga `train_embedding.npz` i `test_embedding.npz`).

## Szacowany czas

| Sprzęt | 1 epoka | Całość (kroki 4–7) |
|---|---|---|
| CPU (i7-12700H, zmierzone) | ok. 17 s | ok. 5–8 h |
| GPU NVIDIA (szacunek) | kilka razy szybciej | ok. 1–2 h |

Na samym CPU można skrócić ablację do jednego seeda (`python ablation.py --seeds 42`),
wtedy w tabeli odchylenie wyjdzie 0.

## Uwagi dla agenta

- Nie zmieniaj kodu modeli ani metryk bez potrzeby. Celem jest przeliczenie wyników.
- Jeśli coś się wysypie, popraw błąd, opisz poprawkę w commicie i uruchom krok ponownie.
- Komendy treningowe uruchamiaj w tle i sprawdzaj logi, bo trwają długo.
- Nie wpisuj do README liczb, których nie ma w plikach `results/`.
