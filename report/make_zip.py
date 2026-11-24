"""Archiwum ZIP do oddania: kod, dane, wyniki, embeddingi, checkpointy, raport.

Uruchom z katalogu glownego repo:  python report/make_zip.py
Wynik: report/VR_biometria_kod_i_materialy.zip
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "VR_biometria_kod_i_materialy.zip"

INCLUDE = [
    "*.py",
    "requirements.txt",
    "README.md",
    "TRENING.md",
    "models/*.py",
    "models/checkpoints/*.pt",
    "Dataset/MotionsNumpy/*.npy",
    "EDA/EDA.ipynb",
    "EDA/figures/*.png",
    "results/*",
    "embeddings/*/*.npz",
    "report/raport.tex",
    "report/raport.pdf",
    "report/sota.tex",
    "report/wnioski.tex",
    "report/make_*.py",
    "report/tables/*.tex",
    "report/figures/*.pdf",
    "report/figures/*.png",
]


def main():
    files = sorted({p for pattern in INCLUDE for p in ROOT.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, Path("VR_biometria") / path.relative_to(ROOT))
    print(f"{OUT.name}: {len(files)} plikow, {OUT.stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
