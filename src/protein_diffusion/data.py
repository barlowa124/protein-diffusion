"""Fetch the GB1 landscape (FLIP mirror of Wu et al., eLife 2016) and split
the training set: only variants measured above config.train_fitness_min.

The diffusion model learns the *functional* region, not the landscape as a
whole — the claim under test is that generated variants land in that region
more often than chance.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from protein_diffusion.config import load_config


def fetch_raw(url: str, out_zip: str) -> Path:
    out = Path(out_zip)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        urllib.request.urlretrieve(url, out)
    return out


def parse_landscape(zip_path: str, variant_col: str, fitness_col: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise RuntimeError(f"expected one CSV in {zip_path}, got {names}")
        with z.open(names[0]) as f:
            df = pd.read_csv(f, low_memory=False)
    for col in (variant_col, fitness_col):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r}: {list(df.columns)}")
    out = df[[variant_col, fitness_col]].dropna().drop_duplicates(variant_col)
    out.columns = ["variant", "fitness"]
    out["fitness"] = out["fitness"].astype(float)
    canonical = out["variant"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{4}")
    n_dropped = int((~canonical).sum())
    if n_dropped:
        print(f"dropping {n_dropped} non-canonical variants (stop/ambiguous)")
    return out[canonical].reset_index(drop=True)


def main(zip_path: str, out_parquet: str):
    cfg = load_config()
    fetch_raw(cfg["dataset"]["url"], zip_path)
    df = parse_landscape(
        zip_path, cfg["dataset"]["variant_col"], cfg["dataset"]["fitness_col"]
    )
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    n_train = int((df.fitness >= cfg["dataset"]["train_fitness_min"]).sum())
    print(
        f"landscape: {len(df)} variants | training pool "
        f"(fitness>={cfg['dataset']['train_fitness_min']}): {n_train}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
