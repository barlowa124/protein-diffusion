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

## Result (seed 7, committed in `results/summary.json`)

| | DDPM generated (n=512) | Random landscape draws (20 seeds) |
|---|---|---|
| mean fitness | **1.10** | 0.081 |
| median fitness | 0.34 | 0.0035 |
| fraction >= 1.0 | **36%** | 2.5% |
| fraction >= 0.5 | 46% | 4.1% |
| true top-100 hits | **11** | 0.3 mean |
| unique variants | 489/512 (95%) | - |
| memorized (in training set) | 45/512 (8.8%) | ~4% expected |
| unmeasured by original screen | 1.0% | - |

Reading it honestly: generation is ~13x enriched over random draws and
**mostly generalization, not memorization** — 467 of 512 samples are novel
variants the model never saw, and they still land in the functional region
far above chance. 5 of 512 samples fell outside the measured set entirely;
they're counted as zero-fitness wasted experiments rather than dropped.

## Caveats

- Four-site combinatorial space is small and discrete; this demonstrates
  generative *mechanics + oracle evaluation*, not full-sequence design.
- GB1's landscape is smooth enough that a much simpler proposal (mutate
  high-fitness parents) might do comparably — the honest baseline to add
  next. The current baseline is uniform random, the minimal counterfactual.
- DDPM in continuous one-hot space is a modeling convenience; discrete
  diffusion over residues is the more principled formulation.
- Single seeded run; generation statistics are one trajectory.

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
