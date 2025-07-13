import torch
import torch.nn as nn
from torch import Tensor
import math


class SinusoidalPE(nn.Module):
    r"""Sinusodial position embedder.
    """
    def __init__(
        self, 
        dim: int = 256, 
        max_period: int = 10000,
        scale: float = 1.0
    ):
        super().__init__()

        # emb = sin(t/10000^(2i/d)) = sin(t·e^{(-2i/d) * ln10000}), i=[0,1, ...,d/2]
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half)  # (b,), e^{(-2i/d) * ln10000}
        self.register_buffer("freqs", freqs)

        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate time embedding.

        b: batch_size
        d: dim

        Args:
            x: (b,), between 0. and 1.

        Outputs:
            out: (b, d)
        """
        
        x = self.scale * torch.outer(x, self.freqs)  # (b, d/2)
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)  # (b, d)
        
        return x


def plot(x, path, title):
    fig, ax = plt.subplots()
    ax.matshow(emb.T, origin='lower', aspect='auto', cmap='jet')
    ax.set_title(title)
    ax.set_xlabel("input")
    ax.set_ylabel("dim")
    ax.xaxis.set_ticks([])
    ax.xaxis.tick_bottom()
    plt.savefig(path)
    print(f"Write out to {path}")


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Example 1: token [0, N] positional embedding (PE)
    x = torch.arange(500)  # (b,)
    pe = SinusoidalPE(dim=256)
    emb = pe(x)  # (b, d)
    plot(emb, "out1.pdf", "token [0, 500] positional embedding")
    
    # Example 2: time [0 ~ 1] positional embedding
    x = torch.arange(0, 1, 0.002)  # (b,)
    pe = SinusoidalPE(dim=256, scale=100.)
    emb = pe(x)  # (b, d)
    plot(emb, "out2.pdf", "time [0 ~ 1] positional embedding")

    # Example 3: spatial [-10 ~ 10] positional embedding
    x = torch.arange(-10, 10, 0.04)  # (b,)
    pe = SinusoidalPE(dim=256, scale=10.)
    emb = pe(x)  # (b, d)
    plot(emb, "out3.pdf", "spatial [-10, 10] positional embedding")

    # Example 4: angle [-π ~ π] positional embedding
    x = torch.arange(-3.14, 3.14, 0.01)  # (b,)
    pe = SinusoidalPE(dim=256, scale=100.)
    emb = pe(x)  # (b, d)
    plot(emb, "out4.pdf", "angle [-π ~ π] PE")
