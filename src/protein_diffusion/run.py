"""Train the DDPM on measured-functional GB1 variants, generate candidates,
and evaluate them against the landscape oracle vs random draws.

Evaluation, all measured not assumed:
- fitness of generated variants (the oracle answers what the experiment
  would have returned)
- fraction unmeasured by the original screen (model can propose variants
  the landscape lacks — reported, not hidden)
- memorization: fraction of generated variants that were literally in the
  training set vs novel high-fitness generalization
- random baseline: uniform draws from the landscape, replicated over seeds
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from protein_diffusion.config import load_config
from protein_diffusion.ddpm import sample, train
from protein_diffusion.encode import decode, one_hot
from protein_diffusion.provenance import write_manifest


def _stats(fitness: np.ndarray, top_set: set, idx: np.ndarray, train_set: set):
    return {
        "n": int(len(fitness)),
        "fitness_mean": float(np.nanmean(fitness)),
        "fitness_median": float(np.nanmedian(fitness)),
        "frac_ge_1": float(np.mean(fitness >= 1.0)),
        "frac_ge_05": float(np.mean(fitness >= 0.5)),
        "top_hits": int(sum(i in top_set for i in idx)),
        "n_memorized": int(sum(i in train_set for i in idx)),
    }


def main(in_parquet: str, out_json: str, out_model: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    fit_map = dict(zip(df["variant"], df["fitness"]))
    train_mask = df["fitness"] >= cfg["dataset"]["train_fitness_min"]
    train_df = df[train_mask].reset_index(drop=True)
    # df indices of training rows — NOT train_df.index (those are positions
    # in the filtered frame); gen_idx below indexes the full df
    train_set = set(df.index[train_mask].tolist())
    X = one_hot(train_df["variant"])
    y_all = df["fitness"].to_numpy()
    top_set = set(np.argsort(-y_all)[: cfg["evaluation"]["top_k"]].tolist())
    df_index = {v: i for i, v in enumerate(df["variant"])}

    m = cfg["model"]
    model = train(X, m["timesteps"], m["hidden"], m["lr"], m["epochs"],
                  m["batch_size"], m["seed"])
    import torch

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_model)

    # replicate sampling over seeds: the trained model is fixed, the
    # ancestral sampling chain is what varies — same honesty standard as
    # the random baseline
    n_gen = cfg["generate"].get("n_seeds", 1)
    gen_stats = []
    per_seed_variants = []
    for s in range(n_gen):
        gv = decode(
            sample(model, cfg["generate"]["n_samples"], X.shape[1],
                   m["timesteps"], m["seed"] + 31 * s)
        )
        gi = np.array([df_index.get(v, -1) for v in gv])
        meas = gi >= 0
        gf = np.array(
            [fit_map[v] if i >= 0 else np.nan for v, i in zip(gv, gi)]
        )
        st = {
            **_stats(np.nan_to_num(gf, nan=0.0), top_set,
                     gi[meas], train_set),
            "n_unique": len(set(gv)),
            "frac_unmeasured": float(1 - meas.mean()),
        }
        # the honest split: is elevated fitness real generalization, or
        # just memorized training rows scoring well again?
        mem = np.isin(gi, list(train_set)) & meas
        novel = meas & ~mem
        st["n_memorized"] = int(mem.sum())
        st["fitness_mean_memorized"] = float(np.nanmean(gf[mem])) if mem.any() else None
        st["fitness_mean_novel"] = float(np.nanmean(gf[novel])) if novel.any() else None
        st["frac_ge_1_novel"] = float(np.mean(gf[novel] >= 1.0)) if novel.any() else None
        gen_stats.append(st)
        per_seed_variants.append(gv)

    gen_mean = {
        k: float(np.mean([s[k] for s in gen_stats if s[k] is not None]))
        for k in gen_stats[0]
    }
    gen_std = {
        f"{k}_std": float(np.std([s[k] for s in gen_stats if s[k] is not None]))
        for k in gen_stats[0]
    }

    rng_stats = []
    for s in range(cfg["evaluation"]["n_random_seeds"]):
        r = np.random.default_rng(m["seed"] + 100 + s)
        ri = r.choice(len(df), size=cfg["generate"]["n_samples"], replace=False)
        rng_stats.append(
            _stats(y_all[ri], top_set, ri, train_set)
        )
    rand_mean = {k: float(np.mean([s[k] for s in rng_stats]))
                 for k in rng_stats[0]}

    result = {
        "config": {
            "train_fitness_min": cfg["dataset"]["train_fitness_min"],
            "n_train": int(len(train_df)),
            "timesteps": m["timesteps"],
            "epochs": m["epochs"],
            "n_samples": cfg["generate"]["n_samples"],
        },
        "generated": {
            "n_seeds": n_gen,
            "mean": gen_mean,
            "std": gen_std,
            "per_seed": gen_stats,
        },
        "random": {"n_seeds": len(rng_stats), "mean": rand_mean},
    }
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    write_manifest("results/provenance.json", inputs=[in_parquet])
    print(
        f"generated ({n_gen} seeds): mean fitness "
        f"{gen_mean['fitness_mean']:.3f}±{gen_std['fitness_mean_std']:.3f} "
        f"vs random {rand_mean['fitness_mean']:.3f} | "
        f"memorized {gen_mean['n_memorized']:.1f}/"
        f"{gen_mean['n']} | "
        f"unmeasured {gen_mean['frac_unmeasured']:.2%}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
