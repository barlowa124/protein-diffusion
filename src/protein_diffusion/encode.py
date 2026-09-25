"""One-hot encode/decode for the 4-site combinatorial variant space.

The DDPM operates on the continuous relaxation of the (4, 20) one-hot
tensor; generation decodes by per-site argmax, which always yields a valid
variant string by construction.
"""

import numpy as np
import pandas as pd

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
N_SITES = 4


def one_hot(variants: pd.Series, alphabet: str = AA_ALPHABET) -> np.ndarray:
    """(n, n_sites*20) float32 one-hot."""
    idx = {aa: i for i, aa in enumerate(alphabet)}
    X = np.zeros((len(variants), N_SITES * len(alphabet)), dtype=np.float32)
    for i, v in enumerate(variants):
        if len(v) != N_SITES:
            raise ValueError(f"variant {v!r} is not {N_SITES} residues")
        for site, aa in enumerate(v):
            j = idx.get(aa)
            if j is None:
                raise ValueError(f"non-standard residue {aa!r} in {v!r}")
            X[i, site * len(alphabet) + j] = 1.0
    return X


def decode(X: np.ndarray, alphabet: str = AA_ALPHABET) -> list:
    """Per-site argmax over the (n, 4, 20) logits -> list of variant strings."""
    X = np.asarray(X).reshape(-1, N_SITES, len(alphabet))
    return [
        "".join(alphabet[j] for j in row.argmax(axis=1)) for row in X
    ]
