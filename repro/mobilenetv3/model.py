"""MobileNetV3-Small, implemented from scratch for the Milestone I reproduction.

This is a line-by-line realisation of the architecture in Howard et al.,
"Searching for MobileNetV3" (arXiv:1905.02244), Table 2. The goal is a faithful
reproduction, so every design choice in the paper is reflected here and tagged
with the section it comes from:

  * Inverted-residual "bneck" block with a linear bottleneck  (V2, kept in V3)
  * Squeeze-and-Excite set to 1/4 of the *expansion* channels (section 5.3)
  * h-swish / h-sigmoid nonlinearities                         (section 5.2)
  * h-swish used only in the second half of the network        (section 5.2)
  * The efficient last stage: 1x1 conv -> pool -> 1x1 conv     (section 5.1)

NOTHING is imported from timm or torchvision here on purpose - the whole point
of the exercise is that the network is built from torch.nn primitives. The
companion script verify_architecture.py checks this module against
torchvision's reference implementation to prove the reproduction is correct.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Nonlinearities (paper section 5.2)
# ----------------------------------------------------------------------------
class HSigmoid(nn.Module):
    """Hard sigmoid: ReLU6(x + 3) / 6.

    The paper replaces the sigmoid in both the SE gate and swish with this
    piece-wise-linear form because it is cheaper and quantization-friendly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Hard swish: x * ReLU6(x + 3) / 6  (paper section 5.2)."""

    def __init__(self) -> None:
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.relu6(x + 3.0) / 6.0)


def _make_divisible(value: float, divisor: int = 8) -> int:
    """Round channel counts to a multiple of 8.

    Every MobileNet keeps channel counts divisible by 8 so the tensors map
    cleanly onto hardware. The paper's channel widths are already divisible, so
    at multiplier 1.0 this is a no-op, but it is needed for other multipliers
    and matches the reference implementations.
    """
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    # Rounding down must not drop more than 10% of the channels.
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


# ----------------------------------------------------------------------------
# Squeeze-and-Excite (paper section 5.3)
# ----------------------------------------------------------------------------
class SqueezeExcite(nn.Module):
    """SE module with the reduction fixed to 1/4 of the expansion channels.

    Section 5.3: "we replace them all to fixed to be 1/4 of the number of
    channels in expansion layer." The reduced dimension is kept divisible by 8.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        reduced = _make_divisible(channels // 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.gate = HSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.gate(self.fc2(scale))
        return x * scale


# ----------------------------------------------------------------------------
# Inverted-residual bottleneck ("bneck")
# ----------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    """One row of Table 2.

    Expansion (1x1) -> depthwise (kxk) -> optional SE -> projection (1x1).
    A residual connection is added only when stride == 1 and in/out channels
    match (the linear-bottleneck rule inherited from MobileNetV2).
    """

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hswish: bool,
    ) -> None:
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        activation = HSwish if use_hswish else nn.ReLU

        layers: list[nn.Module] = []

        # 1x1 expansion. The first block in the table has exp == in, so the
        # reference implementations skip this conv when there is nothing to
        # expand; we do the same to keep the parameter count exact.
        if exp_channels != in_channels:
            layers += [
                nn.Conv2d(in_channels, exp_channels, 1, bias=False),
                nn.BatchNorm2d(exp_channels),
                activation(inplace=True) if activation is nn.ReLU else activation(),
            ]

        # Depthwise kxk.
        padding = kernel_size // 2
        layers += [
            nn.Conv2d(
                exp_channels,
                exp_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=exp_channels,
                bias=False,
            ),
            nn.BatchNorm2d(exp_channels),
        ]
        if use_se:
            layers.append(SqueezeExcite(exp_channels))
        layers.append(activation(inplace=True) if activation is nn.ReLU else activation())

        # 1x1 projection to the linear bottleneck (no activation).
        layers += [
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ----------------------------------------------------------------------------
# Full network
# ----------------------------------------------------------------------------
class MobileNetV3Small(nn.Module):
    """MobileNetV3-Small (Table 2), parameterised by width multiplier.

    Args:
        num_classes: size of the classification head.
        width_mult:  depth multiplier applied to every channel count (1.0 =
                     the exact table).
        dropout:     dropout before the final classifier (paper uses 0.8 keep
                     prob on ImageNet; expose it so smaller datasets can lower
                     it).
    """

    # (kernel, exp_size, out, use_se, use_hswish, stride) - exactly Table 2.
    # h-swish ("HS") turns on from the 5x5/40 block onward, matching the paper's
    # "second half of the model" rule from section 5.2.
    CONFIG = [
        (3, 16, 16, True, False, 2),
        (3, 72, 24, False, False, 2),
        (3, 88, 24, False, False, 1),
        (5, 96, 40, True, True, 2),
        (5, 240, 40, True, True, 1),
        (5, 240, 40, True, True, 1),
        (5, 120, 48, True, True, 1),
        (5, 144, 48, True, True, 1),
        (5, 288, 96, True, True, 2),
        (5, 576, 96, True, True, 1),
        (5, 576, 96, True, True, 1),
    ]

    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        def c(ch: int) -> int:
            return _make_divisible(ch * width_mult)

        # Stem: 3x3 conv, stride 2, 16 filters, h-swish.
        stem_out = c(16)
        features: list[nn.Module] = [
            nn.Conv2d(3, stem_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out),
            HSwish(),
        ]

        in_ch = stem_out
        for kernel, exp, out, use_se, use_hswish, stride in self.CONFIG:
            exp_ch = c(exp)
            out_ch = c(out)
            features.append(
                InvertedResidual(in_ch, exp_ch, out_ch, kernel, stride, use_se, use_hswish)
            )
            in_ch = out_ch

        # First half of the efficient last stage (section 5.1): a 1x1 conv that
        # expands the final bottleneck (96 -> 576) before pooling.
        last_conv_out = c(576)
        features += [
            nn.Conv2d(in_ch, last_conv_out, 1, bias=False),
            nn.BatchNorm2d(last_conv_out),
            HSwish(),
        ]
        self.features = nn.Sequential(*features)

        # Pool, then the second 1x1 conv (576 -> 1024) placed *after* pooling so
        # it runs at 1x1 spatial resolution - the whole point of section 5.1.
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(last_conv_out, 1024),
            HSwish(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(1024, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2) -> MobileNetV3Small:
    """Factory used by the training and verification scripts."""
    return MobileNetV3Small(num_classes=num_classes, width_mult=width_mult, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Quick self-check: shapes and parameter count.
    net = build_model(num_classes=1000)
    dummy = torch.randn(2, 3, 224, 224)
    out = net(dummy)
    params = count_parameters(net)
    print(f"output shape: {tuple(out.shape)}")
    print(f"parameters:   {params:,} ({params / 1e6:.2f}M)")
    print("paper Table 3: 2.5M for MobileNetV3-Small 1.0")
