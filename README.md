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

|            |              |
|------------|--------------|
| <img width="1544" height="1080" alt="token" src="https://github.com/user-attachments/assets/b5dbab22-af8d-45b0-aa5a-7328166fdcd7" /> |  <img width="1480" height="1074" alt="time" src="https://github.com/user-attachments/assets/10ab9bbc-c4c3-4f5a-9037-09dd56c2fe60" /> |
| <img width="1480" height="1066" alt="position" src="https://github.com/user-attachments/assets/3e04209d-8c40-48a1-a7f7-2b40c0c94ab6" /> | <img width="1486" height="1078" alt="angle" src="https://github.com/user-attachments/assets/347c7599-0405-4ac3-9def-0e51c2fca4dd" /> |

## Cite

```bibtex
@misc{audioflow2025,
  author       = {Qiuqiang Kong},
  title        = {SinusodialPositionalEmbedding},
  year         = {2025},
  howpublished = {\url{https://github.com/qiuqiangkong/sinusoidal_positional_embedding}},
}
```