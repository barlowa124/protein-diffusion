import numpy as np
import pandas as pd
import pytest
import torch

from protein_diffusion.ddpm import (
    Denoiser,
    make_schedule,
    q_sample,
    sample,
    timestep_embedding,
    train,
)
from protein_diffusion.encode import AA_ALPHABET, N_SITES, decode, one_hot


def test_schedule_properties():
    betas, alphas, ab = make_schedule(100)
    assert len(betas) == 100
    assert torch.all(betas > 0) and torch.all(betas < 1)
    assert torch.allclose(alphas, 1 - betas)
    assert ab[0] > ab[-1] > 0  # alpha_bar decays toward 0


def test_q_sample_noise_schedule():
    x0 = torch.ones(4, 8)
    _, _, ab = make_schedule(300)
    zero = torch.zeros(4, 8)
    xt_early, _ = q_sample(
        x0, torch.zeros(4, dtype=torch.long), ab, noise=zero
    )
    xt_late, _ = q_sample(
        x0, torch.full((4,), 299, dtype=torch.long), ab, noise=zero
    )
    # zero noise: x_t = sqrt(alpha_bar_t) * x0 exactly; alpha_bar decays
    # monotonically so late-t signal is attenuated but not erased at T=300
    assert torch.allclose(xt_early, torch.sqrt(ab[0]) * x0)
    assert torch.allclose(xt_late, torch.sqrt(ab[-1]) * x0)
    assert xt_late.abs().max() < x0.max()


def test_timestep_embedding_shape():
    emb = timestep_embedding(torch.arange(5), dim=32)
    assert emb.shape == (5, 32)


def test_denoiser_output_shape():
    m = Denoiser(x_dim=80, hidden=64)
    out = m(torch.randn(3, 80), torch.tensor([0, 50, 99]))
    assert out.shape == (3, 80)


def test_sample_deterministic_and_valid_decode():
    m = Denoiser(x_dim=N_SITES * len(AA_ALPHABET), hidden=32)
    s1 = sample(m, 8, N_SITES * len(AA_ALPHABET), 10, seed=1)
    s2 = sample(m, 8, N_SITES * len(AA_ALPHABET), 10, seed=1)
    assert np.array_equal(s1, s2)
    variants = decode(s1)
    assert len(variants) == 8
    assert all(len(v) == 4 and set(v) <= set(AA_ALPHABET) for v in variants)


def test_encode_decode_roundtrip():
    variants = pd.Series(["VDGV", "AAAA", "WWWW"])
    assert decode(one_hot(variants)) == ["VDGV", "AAAA", "WWWW"]


def test_train_smoke_returns_eval_model():
    X = one_hot(pd.Series(["AAAA", "AAAV", "AAVA", "VAAA"] * 16))
    model = train(
        X, timesteps=20, hidden=32, lr=1e-3, epochs=2,
        batch_size=16, seed=0,
    )
    assert not model.training
