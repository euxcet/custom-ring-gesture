import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bottleneck_dim, bottleneck_dropout=0.0):
        super().__init__()
        # temporal attention
        self.depthwise_temp = nn.Conv1d(in_channels, in_channels, kernel_size, 1, 0, groups=in_channels)
        self.depthwise_attn = nn.Conv1d(in_channels, in_channels, kernel_size, 1, 0, groups=in_channels)
        
        # channel attention
        self.bottleneck_fc = nn.Sequential(
            nn.Conv1d(in_channels, bottleneck_dim, 1, 1),
            nn.Dropout(bottleneck_dropout),  # improves performance on OPPORTUNITY
            nn.SiLU(inplace=True),
            nn.Conv1d(bottleneck_dim, out_channels, 1, 1),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        y, _ = self.get_gate(x)
        return x * y

    def get_gate(self, x):
        # temporal attention
        y = self.depthwise_attn(x)                                                 # B, C, L
        y = torch.mean(y, dim=1, keepdim=True)                                     # B, 1, L
        attn = F.softmax(y, dim=2)                                                 # B, 1, L
        y = torch.sum(F.silu(self.depthwise_temp(x)) * attn, dim=2, keepdim=True)  # B, C, 1

        # channel attention
        y = self.bottleneck_fc(y)
        return y, attn

class ChannelwiseFilter(nn.Module):
    """
        Applies N_FILTERS (1, KERNEL_SIZE) filters over each channel. All channels share the same kernel weights.
        Implemented as Conv2D on an input of size B * 1 * C * L with N_FILTERS channels and (1, KERNEL_SIZE) kernels.
        Input: B * C * L
        Output: B * (C * N_FILTERS) * L
    """
    def __init__(self, in_channels, n_filters, kernel_size, norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.n_filters = n_filters
        self.channelwise_filters = nn.Conv2d(1, n_filters, (1, kernel_size), stride=1,
                                             padding=(0, kernel_size // 2), bias=not norm)
        self.norm = nn.BatchNorm1d(in_channels * n_filters) if norm else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x = x.unsqueeze(1)                                                # B, 1, C, L
        x = self.channelwise_filters(x)                                   # B, N, C, L
        x = x.view(B, self.in_channels * self.n_filters, L).contiguous()  # B, N * C, L
        x = self.norm(x)
        x = F.silu(x)
        return x

class DepthwiseSeparableConv1d(nn.Module):
    """
        This is used in MobileNet to reduce the number of parameters
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, norm=True, depthwise_multiplier=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels * depthwise_multiplier, kernel_size, stride, padding, groups=in_channels, bias=not norm)
        self.pointwise = nn.Conv1d(in_channels * depthwise_multiplier, out_channels, 1, 1, bias=not norm)
        self.norm = nn.BatchNorm1d(out_channels) if norm else nn.Identity()
    
    def forward(self, x):
        x = F.silu(self.depthwise(x))
        x = F.silu(self.norm(self.pointwise(x)))
        return x
    
class SenseLiteBlockV2(nn.Module):
    """
        SenseLite Block V2
    """
    def __init__(self, in_channels, out_channels, kernel_size, norm=True, sca_reduction=4, dropout=0.0):
        super().__init__()
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1, 1)
        self.conv = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, norm=norm, depthwise_multiplier=2),
            nn.Dropout1d(dropout),
            DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, norm=norm, depthwise_multiplier=2),
        )
        self.sca = SeparableChannelAttention(in_channels + out_channels, out_channels, kernel_size, out_channels // sca_reduction, dropout)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        residual = self.skip_connection(x)
        x = self.conv(x)
        gating_vector, _ = self.sca.get_gate(torch.cat((residual, x), dim=1))
        x = x * gating_vector
        x = x + residual
        x = self.dropout(x)  # dropout here does not interfere with pooling, since nn.Dropout1D drops entire channels
        return x

class SenseLiteV2(nn.Module):
    """
        SenseLite V2
    """
    def __init__(self, input_len: int, input_dim: int, output_dim: int, conv_dim: list[int], dropout: float, fc_dim: int, sca_reduction: int, n_input_filters: int):
        super().__init__()
        # input block with channelwise filters
        self.input_block = nn.Sequential(
            nn.Dropout1d(dropout),  # DRIP
            ChannelwiseFilter(input_dim, n_input_filters, kernel_size=5, norm=False),
            nn.Dropout1d(dropout),
            nn.Conv1d(input_dim * n_input_filters, conv_dim[0], 1, 1),
            nn.BatchNorm1d(conv_dim[0]),
            nn.SiLU(inplace=True),
            nn.AvgPool1d(2, 2),
            nn.Dropout1d(dropout),
        )
        input_len //= 2

        # SenseLite backbone
        self.backbone = nn.ModuleList()
        for block_idx, (in_channels, out_channels) in enumerate(zip(conv_dim[: -1], conv_dim[1: ])):
            self.backbone.append(SenseLiteBlockV2(in_channels, out_channels, 5, norm=True, sca_reduction=sca_reduction, dropout=dropout))
            if block_idx != len(conv_dim) - 2:
                self.backbone.append(nn.AvgPool1d(2, 2))
                input_len //= 2
            else:
                self.backbone.append(nn.AdaptiveAvgPool1d(1))

        # classification head
        self.fc = nn.Sequential(
            nn.Linear(conv_dim[-1], fc_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim, output_dim)
        )
    
    def forward(self, x):
        # input block
        x = self.input_block(x)

        # SenseLite backbone
        for module in self.backbone:
            x = module(x)
        x = x.squeeze(2)  # B, C, L -> B, C

        # classifier
        x = self.fc(x)
        return x


class SenseLiteV2Custom(nn.Module):
    """
        SenseLite V2
    """
    def __init__(self, input_len: int, input_dim: int, output_dim: int, conv_dim: list[int], dropout: float, fc_dim: int, sca_reduction: int, n_input_filters: int):
        super().__init__()
        # input block with channelwise filters
        self.input_block = nn.Sequential(
            nn.Dropout1d(dropout),  # DRIP
            ChannelwiseFilter(input_dim, n_input_filters, kernel_size=5, norm=False),
            nn.Dropout1d(dropout),
            nn.Conv1d(input_dim * n_input_filters, conv_dim[0], 1, 1),
            nn.BatchNorm1d(conv_dim[0]),
            nn.SiLU(inplace=True),
            nn.AvgPool1d(2, 2),
            nn.Dropout1d(dropout),
        )
        input_len //= 2

        # SenseLite backbone
        self.backbone = nn.ModuleList()
        for block_idx, (in_channels, out_channels) in enumerate(zip(conv_dim[: -1], conv_dim[1: ])):
            self.backbone.append(SenseLiteBlockV2(in_channels, out_channels, 5, norm=True, sca_reduction=sca_reduction, dropout=dropout))
            if block_idx != len(conv_dim) - 2:
                self.backbone.append(nn.AvgPool1d(2, 2))
                input_len //= 2
            else:
                self.backbone.append(nn.AdaptiveAvgPool1d(1))

        # classification head
        self.fc = nn.Sequential(
            nn.Linear(conv_dim[-1], fc_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_dim, output_dim)
        )
    
    def forward(self, x):
        # input block
        x = self.input_block(x)

        # SenseLite backbone
        for module in self.backbone:
            x = module(x)
        x = x.squeeze(2)  # B, C, L -> B, C

        # classifier
        x = self.fc(x)
        return x
