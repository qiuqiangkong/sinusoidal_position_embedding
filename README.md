# Sinusoidal Positional Embedding

This is a python implementation of sinusoidal positional embedding. 

## Usage

# Example 1: token [0, N] positional embedding (PE)

```python
import torch
from sinusoidal_pe import SinusoidalPE

x = torch.arange(500)  # (b,)
pe = SinusoidalPE(dim=256)
emb = pe(x)  # (b, d)
```

# Example 2: time [0 ~ 1] positional embedding

```python
x = torch.arange(0, 1, 0.01)  # (b,)
pe = SinusoidalPE(dim=256, scale=100.)
emb = pe(x)  # (b, d)
```

# Example 3: spatial [-w ~ w] positional embedding

```python
x = torch.arange(-10, 10, 0.1)  # (b,)
pe = SinusoidalPE(dim=256, scale=10.)
emb = pe(x)  # (b, d)
```

# Example 4: angle [-π ~ π] positional embedding

```python
x = torch.arange(-3.14, 3.14, 0.01)  # (b,)
pe = SinusoidalPE(dim=256, scale=100.)
emb = pe(x)  # (b, d)
```

See more examples by running the follow command.

```python
python sinusoidal_pe.py
```

## Tips

Tune the `scale` argument so that the positional embedding curve is neither too smooth nor too sharp.

## Visualization of positional embedding

| |      |
| |      |
| |      |
| |      |
