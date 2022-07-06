from math import floor
from torch.nn import Linear, Sequential, BatchNorm2d, Dropout
from .layer import Transition, Dense, GlobalPool2d


def DenseNet(
    num_classes, in_channels, k_channels=32, num_dense=(6, 12, 24, 16), compression=0.5
):
    out_channels = floor(k_channels / compression)
    layers = [
        Transition(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size_conv=7,
            kernel_size_pool=3,
            stride_conv=2,
            stride_pool=2,
        )
    ]

    in_channels = out_channels
    for i, n in enumerate(num_dense):
        layers += [Dense(in_channels=in_channels, k_channels=k_channels, num_layers=n)]
        in_channels += k_channels * n
        out_channels = floor(in_channels * compression)
        if i < len(num_dense) - 1:
            layers += [
                Transition(
                    in_channels=in_channels, out_channels=out_channels, pool="avg"
                )
            ]
            in_channels = out_channels
    layers += [BatchNorm2d(num_features=in_channels)]
    layers += [GlobalPool2d(pool="avg")]
    layers += [Linear(in_features=in_channels, out_features=num_classes)]

    return Sequential(*layers)


if __name__ == "__main__":
    from torch import rand, no_grad

    with no_grad():
        inputs = rand(32, 3, 224, 224)
        model = DenseNet(1000, 3)
        print(model(inputs).shape)
        print(f"{sum(p.numel() for p in model.parameters()):,}")
