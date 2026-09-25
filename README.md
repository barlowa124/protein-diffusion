# protein-diffusion

A **conditional DDPM** trained on the full measured GB1 fitness landscape
and steered toward high fitness with classifier-free guidance. Generative
modeling where "did it work" is answerable from the measured oracle, not
vibes.

**Status: working demonstration on GB1, documented transfer failure on
AAV.** Snakemake DAG runs fetch -> train -> guided sampling -> oracle
evaluation on the GB1 four-site combinatorial landscape (Wu et al., eLife
2016, via the FLIP mirror): 149,361 variants at sites V39/D40/G41/V54 with
experimentally measured enrichment fitness. A second-landscape attempt on
AAV2 capsid viability (Ogden et al., Science 2019) fails for two
independent, measured reasons (see "Second landscape").

## Design

- **Data space**: continuous relaxation of the 4-site one-hot tensor
  (4 x 20 = 80 dims). Generation decodes by per-site argmax, so every sample
  is a valid variant string by construction.
- **Model**: epsilon-prediction DDPM: linear beta schedule (T=300), 2-layer
  MLP denoiser (hidden 256), ancestral sampling. The denoiser takes a
  fitness condition channel (log1p-scaled). 15% condition dropout during
  training enables classifier-free guidance:
  `eps = eps_uncond + w * (eps_cond - eps_uncond)`.
- **Training set**: the full 149k measured landscape, including the dead
  variants. The v1 fit-only training set is why v1 failed. A model
  that never sees dead variants cannot learn where the boundary is.
- **Evaluation is against the oracle**: generated variants are looked up
  in the *measured* landscape, real experimental fitness, not a surrogate
  scoring its own outputs.

## Result (8 sampling seeds each, committed in `results/summary.json`)

| Sampling mode | mean fitness | >= 0.5 | >= 1.0 | top-100 hits | unique/512 |
|---|---|---|---|---|---|
| unconditioned | 0.061 | 2.7% | 1.7% | 0.2 | 510 |
| conditioned (w=0) | 0.163 | 7.5% | 5.0% | 1.2 | 509 |
| **guided w=4** | **1.04** | **47.2%** | 36.1% | 7.6 | 462 |
| **guided w=8** | **1.69** | **80.4%** | **63.2%** | 6.6 | 237 |
| mutate parent, mu=1 | 1.28 | 45.3% | 39.1% | **16.8** | **500** |
| mutate parent, mu=2 | 0.81 | 30.3% | 25.5% | 8.5 | 506 |
| random draws | 0.081 | 4.1% | 2.5% | 0.3 | — |

The guidance dose-response is the demonstration: conditioning alone shifts
the distribution modestly (7.5% fit), and cranking the CFG weight steers
hard into the functional region. 80% of proposals measure >= 0.5 vs 4%
at random, ~20x enrichment. The tradeoff is in the last columns:
strong guidance concentrates samples onto fewer modes (unique variants
510 -> 237, and top-100 hits dip from 7.6 to 6.6 as the sampler
collapses onto "good enough" modes instead of the very best). Diversity
vs fitness is the classic CFG tradeoff, measured here instead of assumed.

## The mutational baseline

`mutate_*` rows are the trivial experimental baseline a diffusion model
has to justify itself against: draw a parent uniformly from the >=3.0
fitness pool, apply `max(1, Poisson(mu))` random substitutions, score the
children. The verdict is mixed:

- The earlier deconstruction ("Hamming-1 neighbors of fit variants are
  dead, mean 0.087") held for neighbors of *marginally* fit rows (>=0.5).
  The **elite peak is locally smoother**: children of >=3.0 parents
  measure 45% >= 0.5 at mu=1, so mutation is not useless here.
- Guided diffusion still wins on **bulk enrichment**: 80% vs 45% fit rate,
  1.69 vs 1.28 mean fitness. CFG concentrates mass better than random
  mutagenesis.
- On **discovery metrics the trivial baseline wins**: 500 unique
  variants and ~17 top-100 hits per 512 samples vs the DDPM's 237 / 6.6.
  CFG buys hit-rate by spending diversity, and this landscape's elite
  neighborhood is connected enough that mutagenesis rides it.

The claim narrows accordingly: the DDPM demonstrates *conditional
steering at superior hit-rate*, not superiority over all baselines at
every objective.

## Why v1 failed, and what fixed it

The first version trained an *unconditional* DDPM on only the ~5.8k
fitness >= 0.5 variants. It produced 36% fit samples, but a memorization
audit (after fixing an indexing bug that undercounted it as 8.8%) showed
**46% of samples were literal training rows, and novel variants scored at
landscape-random fitness (0.082 vs 0.081, 0% >= 1.0).**

The measured deconstruction:

- Samples were not bit-copies pre-decode (mean continuous distance to the
  nearest training point ~2.1 for both memorized and novel), and memorization
  happened at argmax: samples were blurs snapping to the nearest vertex.
- 90% of novel outputs were Hamming-1 neighbors of training rows. But
  GB1's functional region is an archipelago: Hamming-1 neighbors of fit
  variants measure 0.087 mean and **0% are >= 0.5**. Near-copy sampling is
  worthless on a landscape this sharp.
- Conclusion: an unconditional density model over a sparse fit set cannot
  generalize, since there is no smooth manifold to interpolate along. The fix
  is the *conditioning signal* (fitness, over the full landscape including
  dead variants) and the *mechanism* (guidance) to spend probability mass
  where it counts.

## Second landscape: AAV2, and why it fails twice

Same code, `config/config_aav.yaml`: 28-aa `mutated_region`, alphabet
extended with `*` stop variants (kept, since they are real dead variants),
`shift_log1p` conditioning for the negative log-viability scores
(`results/summary_aav.json`).

**Failure 1: the oracle is vacuous.** GB1 is 93% measured over its
4-site space, so decoded variants almost always have ground truth. AAV
is 38k designed variants inside a ~21^28 region: **100% of generated
samples are unmeasured**. No decoded string coincides with a measured
row, so oracle fitness is undefined for them. This is the GB1
"93% measured" caveat inverted. AAV shows the caveat was
load-bearing.

**Failure 2: the model doesn't even learn the library.** Diagnostics:
the library is dense (median pairwise Hamming 7, nearest-member distance
2) and per-site conserved (median site entropy 0.87 nats vs 3.0 uniform).
Yet generated strings sit ~22 substitutions from every measured variant,
near random-string distance, and match the library's modal residue at
only 9% of sites (library members: 86%). Scaling the denoiser (512 hidden,
40 epochs) improves the match to 29% and MSE 0.64→0.29, so undertraining is
part of it, but the samples remain far off-manifold. Where GB1's
memorization failure produced *plausible-looking* outputs, AAV produces
visible noise. Neither is a working generator, for different reasons.

What a working version would need: a decoder-aware sampler (projected /
discrete diffusion) or a learned-fitness proxy for eval, each with its
own circularity caveats. Documented, not implemented.

## Caveats

- Four-site combinatorial space is small and discrete, and 93% of it is
  measured, so "novel" is nearly unreachable by construction. The claim
  demonstrated is *conditional steering*, not de novo discovery. On a real
  protein the same machinery would need a novelty channel to be useful.
- Diversity collapse at high guidance is real and reported. A proposal
  engine would sweep w to trade hit-rate against diversity.
- DDPM in continuous one-hot space is a modeling convenience. Discrete
  diffusion (D3PM-style) over residues is the principled formulation.
- A simpler proposal distribution (sampling the empirical high-fitness
  pool directly) would trivially produce fit variants. The diffusion
  model is justified only when conditioning needs to generalize, which
  this landscape cannot test.

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
al., eLife 2016 supplement). Downloaded zip is gitignored. The parsed
parquet is a regenerable intermediate.
