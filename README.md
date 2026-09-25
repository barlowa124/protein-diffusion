# protein-diffusion

A **conditional DDPM** trained on the full measured GB1 fitness landscape
and steered toward high fitness with classifier-free guidance — generative
modeling where "did it work" is answerable from the measured oracle, not
vibes.

**Status: working demonstration.** Snakemake DAG runs fetch -> train ->
guided sampling -> oracle evaluation on the GB1 four-site combinatorial
landscape (Wu et al., eLife 2016, via the FLIP mirror): 149,361 variants
at sites V39/D40/G41/V54 with experimentally measured enrichment fitness.

## Design

- **Data space**: continuous relaxation of the 4-site one-hot tensor
  (4 x 20 = 80 dims). Generation decodes by per-site argmax, so every sample
  is a valid variant string by construction.
- **Model**: epsilon-prediction DDPM — linear beta schedule (T=300), 2-layer
  MLP denoiser (hidden 256), ancestral sampling. The denoiser takes a
  fitness condition channel (log1p-scaled); 15% condition dropout during
  training enables classifier-free guidance:
  `eps = eps_uncond + w * (eps_cond - eps_uncond)`.
- **Training set**: the full 149k measured landscape — including the dead
  variants. The v1 fit-only training set is exactly why v1 failed; a model
  that never sees dead variants cannot learn where the boundary is.
- **Evaluation is against the oracle**: generated variants are looked up
  in the *measured* landscape — real experimental fitness, not a surrogate
  scoring its own outputs.

## Result (8 sampling seeds each, committed in `results/summary.json`)

| Sampling mode | mean fitness | >= 0.5 | >= 1.0 | top-100 hits | unique/512 |
|---|---|---|---|---|---|
| unconditioned | 0.061 | 2.7% | 1.7% | 0.2 | 510 |
| conditioned (w=0) | 0.163 | 7.5% | 5.0% | 1.2 | 509 |
| **guided w=4** | **1.04** | **47.2%** | 36.1% | 7.6 | 462 |
| **guided w=8** | **1.69** | **80.4%** | **63.2%** | 6.6 | 237 |
| random draws | 0.081 | 4.1% | 2.5% | 0.3 | — |

The guidance dose-response is the demonstration: conditioning alone shifts
the distribution modestly (7.5% fit), and cranking the CFG weight steers
hard into the functional region — 80% of proposals measure >= 0.5 vs 4%
at random, ~20x enrichment. The honest tradeoff is in the last columns:
strong guidance concentrates samples onto fewer modes (unique variants
510 -> 237, and top-100 hits actually dip from 7.6 to 6.6 as the sampler
collapses onto "good enough" modes rather than the very best). Diversity
vs fitness is the classic CFG tradeoff, measured here rather than assumed.

## Why v1 failed, and what fixed it

The first version trained an *unconditional* DDPM on only the ~5.8k
fitness >= 0.5 variants. It produced 36% fit samples — but a memorization
audit (after fixing an indexing bug that undercounted it as 8.8%) showed
**46% of samples were literal training rows, and novel variants scored at
landscape-random fitness (0.082 vs 0.081, 0% >= 1.0).**

The deconstruction, measured:

- Samples were not bit-copies pre-decode (mean continuous distance to the
  nearest training point ~2.1 for both memorized and novel) — memorization
  happened at argmax: samples were blurs snapping to the nearest vertex.
- 90% of novel outputs were Hamming-1 neighbors of training rows. But
  GB1's functional region is an archipelago — Hamming-1 neighbors of fit
  variants measure 0.087 mean and **0% are >= 0.5**. Near-copy sampling is
  worthless on a landscape this sharp.
- Conclusion: an unconditional density model over a sparse fit set cannot
  generalize — there is no smooth manifold to interpolate along. The fix
  is not "less memorization"; it is giving the model the *conditioning
  signal* (fitness, over the full landscape including dead variants) and
  the *mechanism* (guidance) to spend probability mass deliberately.

## Caveats

- Four-site combinatorial space is small and discrete, and 93% of it is
  measured — so "novel" is nearly unreachable by construction; the claim
  demonstrated is *conditional steering*, not de novo discovery. On a real
  protein the same machinery would need a novelty channel to be useful.
- Diversity collapse at high guidance is real and reported; a proposal
  engine would sweep w to trade hit-rate against diversity.
- DDPM in continuous one-hot space is a modeling convenience; discrete
  diffusion (D3PM-style) over residues is the principled formulation.
- A simpler proposal distribution (mutating fit parents is useless here —
  Hamming-1 is dead; sampling the empirical high-fitness pool directly)
  would trivially produce fit variants — the diffusion model earns its
  keep only when conditioning needs to generalize, which this landscape
  cannot test. Stated plainly.

## Run

```bash
.venv/bin/snakemake -j1          # fetch -> train -> sample -> evaluate
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q
```

`DIFFUSION_CONFIG` env var selects an alternate config. Outputs:
`results/summary.json`, `results/provenance.json`; the model checkpoint and
intermediates live in `data/processed/` (regenerable, gitignored).

## Data

FLIP `splits/gb1/four_mutations_full_data.csv.zip` (CC BY 4.0; extends Wu et
al., eLife 2016 supplement). Downloaded zip is gitignored; the parsed
parquet is a regenerable intermediate.
