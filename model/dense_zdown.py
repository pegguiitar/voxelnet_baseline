"""dense_zdown.py - dense (nn.Conv3d) mirror of zdown_to_sparse2d.ZDownTo2D, for
dense counterparts of experiments whose sparse backbone finishes z-compression via
that shared module (dense_baseline_exp2_bev). No restrict_xy_support-style
filtering needed here -- a dense conv already computes at every (y,x) position, so
there's no "dilation" to correct for; this is just the standard Conv3d-per-stage
stack, matching the sparse version's exact channel/kernel/stride/padding shapes so
dense-vs-sparse compute is the only difference."""
import torch.nn as nn


class DenseZDown(nn.Module):
    def __init__(self, in_channels, stage_channels, kernel_size=3):
        super().__init__()
        layers = []
        c_in = in_channels
        pad = kernel_size // 2
        for c_out in stage_channels:
            layers += [
                nn.Conv3d(c_in, c_out, kernel_size, stride=(2, 1, 1), padding=(pad, pad, pad), bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in

    @staticmethod
    def output_d(d_in: int, num_stages: int, kernel_size: int = 3) -> int:
        d = d_in
        pad = kernel_size // 2
        for _ in range(num_stages):
            d = (d + 2 * pad - kernel_size) // 2 + 1
        return d

    def forward(self, x):
        return self.net(x)
