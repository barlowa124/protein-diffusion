"""Minimal DDPM over the continuous relaxation of variant one-hot vectors.

Standard Gaussian diffusion (Ho et al. 2020 parameterization): linear beta
schedule, epsilon-prediction MLP with sinusoidal timestep embedding,
ancestral reverse sampling. Deliberately small — the data space is 80 dims.
"""

import numpy as np
import torch
import torch.nn as nn


def make_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def timestep_embedding(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal embedding; t in [0, T) integer timesteps."""
    half = dim // 2
    freq = torch.exp(
        -np.log(10000.0) * torch.arange(half, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freq[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class Denoiser(nn.Module):
    def __init__(self, x_dim: int, hidden: int, t_dim: int = 32):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.t_dim).to(x.device)
        return self.net(torch.cat([x, emb], dim=-1))


def q_sample(x0, t, alpha_bar, noise=None):
    """Forward noising: x_t = sqrt(ab_t) x0 + sqrt(1-ab_t) eps."""
    noise = torch.randn_like(x0) if noise is None else noise
    ab = alpha_bar.to(x0.device)[t][:, None]
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise, noise


def train(
    X: np.ndarray,
    timesteps: int,
    hidden: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Denoiser:
    torch.manual_seed(seed)
    betas, alphas, alpha_bar = make_schedule(timesteps)
    x0 = torch.tensor(X, dtype=torch.float32)
    model = Denoiser(x0.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(x0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            xb = x0[perm[i : i + batch_size]]
            t = torch.randint(0, timesteps, (len(xb),))
            xt, eps = q_sample(xb, t, alpha_bar)
            loss = nn.functional.mse_loss(model(xt, t), eps)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if ep % 10 == 9 or ep == 0:
            print(f"epoch {ep + 1}/{epochs} mse {total / n:.4f}")
    model.eval()
    return model


@torch.no_grad()
def sample(model: Denoiser, n: int, x_dim: int, timesteps: int, seed: int):
    """Ancestral DDPM sampling: x_T ~ N(0,I) -> x_0."""
    g = torch.Generator().manual_seed(seed)
    betas, alphas, alpha_bar = make_schedule(timesteps)
    x = torch.randn(n, x_dim, generator=g)
    for t in range(timesteps - 1, -1, -1):
        tt = torch.full((n,), t, dtype=torch.long)
        eps = model(x, tt)
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0)
        coef = (1 - alphas[t]) / torch.sqrt(1 - ab_t)
        mean = (x - coef * eps) / torch.sqrt(alphas[t])
        if t > 0:
            sigma = torch.sqrt(
                (1 - ab_prev) / (1 - ab_t) * betas[t]
            )
            x = mean + sigma * torch.randn(n, x_dim, generator=g)
        else:
            x = mean
    return x.numpy()
