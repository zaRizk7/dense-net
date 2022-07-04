from inspect import Traceback
from multiprocessing import Pool
from torch import Tensor, nn, cat
from typing import Tuple


class GlobalPool2d(nn.Module):
    def __init__(self, pool: str) -> None:
        super().__init__()
        if not pool in ["max", "min", "avg"]:
            raise ValueError("pool must either be 'max', 'min', or 'avg'")
        self.pool = pool

    def forward(self, inputs: Tensor) -> Tensor:
        if self.pool == "max":
            return inputs.max((2, 3))
        elif self.pool == "min":
            return inputs.min((2, 3))
        return inputs.mean((2, 3))


class Dense(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_layers, kernel_size=(1, 3)
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(
                Bottleneck(in_channels + out_channels * l, out_channels, kernel_size)
            )

    def forward(self, inputs):
        for conv in self.convs:
            outputs = conv(inputs)
            inputs = cat([inputs, outputs], 1)
        return inputs


def Conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
) -> nn.Sequential:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd number!")
    padding = 0 if kernel_size > 1 else kernel_size - 2
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    )


def Bottleneck(
    in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (1, 3)
) -> nn.Sequential:
    return nn.Sequential(
        Conv(in_channels, out_channels * 4, kernel_size[0]),
        Conv(out_channels * 4, out_channels, kernel_size[-1]),
    )


def Pool2d(pool: str, kernel_size: int, stride: int) -> nn.Module:
    if pool == "max":
        return nn.MaxPool2d(kernel_size, stride)
    elif pool == "avg":
        return nn.AvgPool2d(kernel_size, stride)
    raise ValueError("pool must either be 'max' or 'avg'")


def Transition(
    in_channels: int,
    out_channels: int,
    kernel_size_conv: int,
    kernel_size_pool: int,
    stride_conv: int = 1,
    stride_pool: int = 1,
    pool: str = "max",
) -> nn.Sequential:
    return nn.Sequential(
        Conv(in_channels, out_channels, kernel_size_conv, stride_conv),
        Pool2d(pool, kernel_size_pool, stride_pool),
    )


if __name__ == "__main__":
    from torch import rand, no_grad

    with no_grad():
        x = rand(32, 64, 224, 224)
        dense = Dense(64, 32, 6)
        print(dense(x).shape)
