PY = ".venv/bin/python"
PP = "PYTHONPATH=src"


rule all:
    input:
        "results/summary.json",


rule fetch:
    output:
        "data/raw/gb1_flip.csv.zip",
    shell:
        "{PP} {PY} -c \"from protein_diffusion.data import fetch_raw; "
        "from protein_diffusion.config import load_config; "
        "fetch_raw(load_config()['dataset']['url'], '{output}')\""


rule prepare:
    input:
        rules.fetch.output,
    output:
        "data/processed/gb1.parquet",
    shell:
        "{PP} {PY} -m protein_diffusion.data {input} {output}"


rule run:
    input:
        rules.prepare.output,
    output:
        summary="results/summary.json",
        model="data/processed/ddpm.pt",
    shell:
        "{PP} {PY} -m protein_diffusion.run {input} {output.summary} {output.model}"
