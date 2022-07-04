from torch import Tensor, nn, cat


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


class Transition(nn.Module):
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
        return outputs


def Conv(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, "same"),
    )


def Bottleneck(
    in_channels: int, out_channels: int, kernel_size=(1, 3)
) -> nn.Sequential:
    return nn.Sequential(
        Conv(in_channels, out_channels * 4, kernel_size[0]),
        Conv(out_channels * 4, out_channels, kernel_size[-1]),
    )


if __name__ == "__main__":
    from torch import rand, no_grad

    with no_grad():
        x = rand(128, 3, 224, 224)
        layer = Transition(3, 32, 6)
        print(layer(x).shape)
