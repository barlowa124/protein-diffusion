"""Inference service: sample variants from a trained DDPM checkpoint.

    MODEL_PATH=data/processed/ddpm.pt uvicorn protein_diffusion.serve:app

Endpoints
    GET  /health      liveness
    GET  /model       checkpoint metadata
    POST /sample      {"n": 32, "seed": 7, "cond": 2.0, "guidance": 4.0}

Fitness conditioning is on the log1p scale used at training time;
`guidance` is the classifier-free w (0 = plain conditional sampling).
Returned `fitness` is the oracle value only when the variant exists in
the landscape — "unmeasured" otherwise, never a surrogate estimate.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from protein_diffusion.ddpm import Denoiser, sample
from protein_diffusion.encode import decode

app = FastAPI(title="protein-diffusion")

_ckpt = os.environ.get("MODEL_PATH", "data/processed/ddpm.pt")
_state = {}


def _load() -> None:
    p = Path(_ckpt)
    sd = torch.load(p, map_location="cpu", weights_only=True)
    x_dim = sd["net.0.weight"].shape[1] - 32 - 1  # minus t_emb, minus cond
    hidden = sd["net.0.weight"].shape[0]
    model = Denoiser(x_dim, hidden)
    model.load_state_dict(sd)
    model.eval()
    timesteps = int(os.environ.get("TIMESTEPS", 300))
    alphabet = os.environ.get("ALPHABET", "ACDEFGHIKLMNPQRSTVWY")
    oracle = None
    lpath = Path(os.environ.get("LANDSCAPE_PATH", "data/processed/gb1.parquet"))
    if lpath.exists():
        ldf = pd.read_parquet(lpath)
        oracle = dict(zip(ldf["variant"], ldf["fitness"]))
    _state.update(model=model, x_dim=x_dim, timesteps=timesteps,
                  alphabet=alphabet, oracle=oracle, path=str(p))


@app.on_event("startup")
def _startup() -> None:
    _load()


class SampleRequest(BaseModel):
    n: int = Field(32, ge=1, le=512)
    seed: int = Field(7, ge=0)
    cond: Optional[float] = None
    guidance: float = Field(0.0, ge=0.0, le=32.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded": "model" in _state}


@app.get("/model")
def model_info() -> dict:
    if "model" not in _state:
        return JSONResponse(status_code=503, content={"error": "not loaded"})
    return {
        "checkpoint": _state["path"],
        "x_dim": _state["x_dim"],
        "timesteps": _state["timesteps"],
        "n_params": int(sum(p.numel() for p in _state["model"].parameters())),
        "oracle_loaded": _state["oracle"] is not None,
    }


@app.post("/sample")
def sample_variants(req: SampleRequest):
    if "model" not in _state:
        return JSONResponse(status_code=503, content={"error": "not loaded"})
    x = sample(_state["model"], req.n, _state["x_dim"], _state["timesteps"],
               req.seed, cond=req.cond, guidance=req.guidance)
    variants = decode(x, _state["alphabet"])
    oracle, out = _state["oracle"], []
    for v in variants:
        measured = oracle.get(v) if oracle else None
        out.append({"variant": v,
                    "fitness": measured if measured is not None else None,
                    "status": "measured" if measured is not None
                              else "unmeasured"})
    return {"n": len(out), "measured": sum(o["status"] == "measured"
                                           for o in out),
            "variants": out}
