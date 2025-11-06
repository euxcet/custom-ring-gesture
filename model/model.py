import torch
from torch import nn

class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=40, bottleneck=True):
        super().__init__()
        ks = [ks // (2**i) for i in range(3)]
        ks = [k if k % 2 != 0 else k - 1 for k in ks]
        self.bottleneck = (
            nn.Conv1d(ni, nf, 1, bias=False) if bottleneck and ni > 1 else nn.Identity()
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(nf if bottleneck else ni, nf, k, padding=k // 2, bias=False)
                for k in ks
            ]
        )
        self.maxconvpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(ni, nf, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(nf * 4)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        conv_outputs = [conv(x) for conv in self.convs]
        maxpool_output = self.maxconvpool(input_tensor)
        x = torch.cat(conv_outputs + [maxpool_output], dim=1)
        return self.act(x)


class InceptionTimeModel(nn.Module):
    def __init__(self, input_channels, num_classes, depth=6):
        super().__init__()
        self.inceptions = nn.ModuleList(
            [
                InceptionModule(input_channels if d == 0 else 32 * 4, 32)
                for d in range(depth)
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32 * 4, num_classes)

    def forward(self, x):
        for inception in self.inceptions:
            x = inception(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class InceptionTimeMoreFcModel(nn.Module):
    def __init__(self, input_channels, num_classes, depth=6):
        super().__init__()
        self.inceptions = nn.ModuleList(
            [
                InceptionModule(input_channels if d == 0 else 32 * 4, 32)
                for d in range(depth)
            ]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        for inception in self.inceptions:
            x = inception(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleModel, self).__init__()
        self.sub = nn.Sequential(
            nn.Flatten(),
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    # 6 * 10 * 20
    def forward(self, x):
        data = []
        for i in range(0, 200, 10):
            data.append(self.sub(x[:, :, i: i + 10]))
        x = torch.concat(data, dim=1)
        x = self.fc(x)
        return x
