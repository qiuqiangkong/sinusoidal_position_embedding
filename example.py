import torch
from torch import Tensor

from sinusoidal_pe import SinusoidalPE


if __name__ == '__main__':

    # Example of encode source and microphone positions and look at directions

    B = 4  # batch_size
    dim = 128

    pos_pe = SinusoidalPE(dim=dim, scale=10.)
    ang_pe = SinusoidalPE(dim=dim, scale=100.)

    # Source
    src_x = torch.rand(B)  # [-l, l]
    src_y = torch.rand(B)  # [-w, w]
    src_z = torch.rand(B)  # [-h ,h]
    src_azi = torch.rand(B)  # [-π ~ π)
    src_ele = torch.rand(B)  # [-π/2 ~ π/2]

    # Microphone
    mic_x = torch.rand(B)  # [-l, l]
    mic_y = torch.rand(B)  # [-w, w]
    mic_z = torch.rand(B)  # [-h ,h]
    mic_azi = torch.rand(B)  # [-π ~ π)
    mic_ele = torch.rand(B)  # [-π/2 ~ π/2]

    emb = torch.cat((
        pos_pe(src_x), pos_pe(src_y), pos_pe(src_z), ang_pe(src_azi), ang_pe(src_ele),
        pos_pe(mic_x), pos_pe(mic_y), pos_pe(mic_z), ang_pe(mic_azi), ang_pe(mic_ele),
    ), dim=-1)  # (b, d*10)

    print(emb.shape)