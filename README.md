# protein-diffusion

A small DDPM trained on **measured-functional** variants of a real protein
fitness landscape, generating candidate variants that are scored against the
measured oracle — generative modeling where "did it work" is answerable from
data, not vibes.

**Status: working demonstration.** Snakemake DAG runs fetch -> train ->
sample -> oracle evaluation on the GB1 four-site combinatorial landscape
(Wu et al., eLife 2016, via the FLIP mirror): 149,361 variants at sites
V39/D40/G41/V54 with experimentally measured enrichment fitness.

## Design

- **Data space**: continuous relaxation of the 4-site one-hot tensor
  (4 x 20 = 80 dims). Generation decodes by per-site argmax, so every sample
  is a valid variant string by construction.
- **Model**: epsilon-prediction DDPM — linear beta schedule (T=300), 2-layer
  MLP denoiser (hidden 256) with sinusoidal timestep embeddings, ancestral
  sampling. Deliberately small; the space is 80-dimensional.
- **Training set**: only variants with measured fitness >= 0.5 (~5.8k of
  149k). The claim under test is that the model learns the functional
  region rather than the landscape as a whole.
- **Evaluation is against the oracle**: every generated variant is looked up
  in the *measured* landscape. No surrogate scoring of generated samples —
  the fitness numbers are what the experiment actually returned.

## Result (8 sampling seeds, committed in `results/summary.json`)

| | DDPM generated (n=512/seed) | Random landscape draws (20 seeds) |
|---|---|---|
| mean fitness | **1.04 ± 0.06** | 0.081 |
| median fitness | 0.35 | 0.0035 |
| fraction >= 1.0 | 36% ± 2% | 2.5% |
| true top-100 hits | 7.8 ± 2.5 | 0.3 mean |
| unique variants | ~488/512 | - |
| **memorized (in training set)** | **238/512 (46%)** | ~4% expected |
| unmeasured by original screen | 0.9% | - |

And the decomposition that matters — splitting generated samples by
whether they copy the training set:

| Subset | Fraction of samples | Mean fitness | >= 1.0 |
|---|---|---|---|
| memorized training rows | ~46% | 2.15 | drives all the enrichment |
| **novel variants** | ~54% | **0.082** | **0%** |
| random landscape draw | — | 0.081 | 2.5% |

The honest headline flipped after a bug fix (see below): **all of the
model's lift is memorization.** Novel generations — the 54% of samples
that are real new variants — sit at exactly landscape-random fitness
(0.082 vs 0.081). A DDPM trained on 5.8k points in an 80-dim space learns
to reproduce its training set, not the fitness landscape's geometry.
That's the well-known diffusion-memorization failure mode, measured end
to end against an oracle rather than assumed away.

## Debugging trail (kept, it's the point)

The first version of this table claimed "mostly generalization, not
memorization" (8.8% memorized). That was a bug: `train_set` was built
from positions in the *filtered* training frame while generated variants
were looked up by *full-frame* index — so a sample was counted memorized
only when its row index happened to be < 5822. The corrected count is
46%, and splitting fitness by membership showed the entire enrichment
came from the memorized half. Any memorization metric is only as good as
its indexing — worth checking before believing the flattering number.

## Caveats

- Four-site combinatorial space is small and discrete; this demonstrates
  generative *mechanics + oracle evaluation + a measured memorization
  failure*, not de novo design capability.
- Fix directions, untested: fewer epochs, a larger/noisier training
  threshold, classifier-free guidance toward fitness, or discrete-state
  diffusion (D3PM-style) over residues.
- GB1's landscape is smooth enough that a simpler proposal (mutating
  high-fitness parents) would likely beat this — uniform random is the
  minimal counterfactual, not a strong one.
- DDPM in continuous one-hot space is a modeling convenience; discrete
  diffusion over residues is the more principled formulation.

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
