import torch
import torch.nn as nn
import torch.nn.functional as F

class WideBasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout_rate=0.0,
    ):

        super().__init__()

        self.bn1 = nn.BatchNorm2d(
            in_channels,
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.dropout = nn.Dropout(
            p=dropout_rate,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):

            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

        else:

            self.shortcut = nn.Identity()

    def forward(
        self,
        x,
    ):

        out = self.bn1(x)

        out = self.relu(out)

        if isinstance(self.shortcut, nn.Identity):
            shortcut = x    
        else:
            shortcut = self.shortcut(out)

        out = self.conv1(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)

        out += shortcut

        return out
    
class WideResNet(nn.Module):

    def __init__(
        self,
        depth,
        widen_factor,
        num_classes=10,
        dropout_rate=0.0,
        model_name="WideResNet",
    ):

        super().__init__()

        self.model_name = model_name

        assert (
            (depth - 4) % 6 == 0
        )

        n = (depth - 4) // 6

        k = widen_factor  

        channels = [
            16,
            16 * k,
            32 * k,
            64 * k,
        ]

        self.in_channels = channels[0]

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.layer1 = self._make_layer(
            WideBasicBlock,
            channels[1],
            n,
            stride=1,
            dropout_rate=dropout_rate,
        )

        self.layer2 = self._make_layer(
            WideBasicBlock,
            channels[2],
            n,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.layer3 = self._make_layer(
            WideBasicBlock,
            channels[3],
            n,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.bn = nn.BatchNorm2d(
            channels[3],
        )

        self.relu = nn.ReLU(
            inplace=True,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            1,
        )

        self.fc = nn.Linear(
            channels[3],
            num_classes,
        )

    def _make_layer(
        self,
        block,
        out_channels,
        blocks,
        stride,
        dropout_rate,
    ):

        layers = []

        layers.append(

            block(
                self.in_channels,
                out_channels,
                stride,
                dropout_rate,
            )

        )

        self.in_channels = out_channels

        for _ in range(
            1,
            blocks,
        ):

            layers.append(

                block(
                    self.in_channels,
                    out_channels,
                    1,
                    dropout_rate,
                )

            )

        return nn.Sequential(
            *layers,
        )
    
    def forward(
        self,
        x,
    ):

        x = self.conv1(x)

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.bn(x)

        x = self.relu(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1,
        )

        x = self.fc(x)

        return x
    


def WRN2810(
    num_classes=10,
):

    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        model_name="WRN-28-10",
    )