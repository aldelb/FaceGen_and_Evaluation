""" Parts of the U-Net model """
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import utils.constants.constants as constants
from torch.autograd import Function

class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, mask_type='A'):
        super(MaskedConv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # Créer un masque pour empêcher la convolution de voir des éléments futurs
        self.register_buffer('mask', torch.ones_like(self.weight.data))
        
        # Appliquer un masque sur le noyau pour les éléments futurs
        self.mask[:, :, kernel_size // 2 + 1:] = 0  # Bloque les éléments futurs
        if mask_type == 'A':  # Si "A", bloque également l'élément présent
            self.mask[:, :, kernel_size // 2] = 0  # Bloque aussi l'élément courant (utile pour certaines architectures)
    
    def forward(self, x):
        # Appliquer le masque avant la convolution
        self.weight.data *= self.mask # Ensures zero's at masked positions
        return super(MaskedConv1d, self).forward(x)
    
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha=1):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MultiHeadAttentionMerge(pl.LightningModule):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttentionMerge, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True, dropout=constants.dropout)
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, combined_embedding):
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(combined_embedding, combined_embedding, combined_embedding)
        x = self.norm(attn_output + combined_embedding)  # Residual connection
        x = self.linear(x)
        return x
    

class CrossAttention(pl.LightningModule):
    def __init__(self, d_model: int, d_cond: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=d_cond,
            vdim=d_cond,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor = None, key_padding_mask: torch.Tensor = None):
        x = self.cross_attn(
            x, cond, cond,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        x = self.dropout(x)
        return x

class FusionAttention(pl.LightningModule):
    def __init__(self, d_audio: int, d_behav: int, d_cond: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_audio (int): Audio embedding dimension.
            d_behav (int): Behavior embedding dimension.
            d_cond (int): Condition embedding dimension after fusion.
            num_heads (int): Number of heads for multihead attention.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn_audio = nn.MultiheadAttention(d_audio, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_behav = nn.MultiheadAttention(d_behav, num_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_audio + d_behav, d_cond)  # Merge audio and behavior dimensions into d_cond
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prev_audio: torch.Tensor, prev_behav: torch.Tensor):
        # Apply self-attention to prev_audio and prev_behav
        attn_audio, _ = self.self_attn_audio(prev_audio, prev_audio, prev_audio)
        attn_behav, _ = self.self_attn_behav(prev_behav, prev_behav, prev_behav)
        
        # Concatenate the two attention results
        fused_cond = torch.cat([attn_audio, attn_behav], dim=-1)
        
        # Pass through a linear layer to merge and reduce the dimension
        fused_cond = self.fc(fused_cond)
        fused_cond = self.dropout(fused_cond)
        
        return fused_cond

# #####################For hubert3#####################
# class AudioEmbedding(pl.LightningModule):
#     def __init__(self, in_channels, out_channels, kernel):
#         super().__init__()
#         self.conv1 = Conv(in_channels, out_channels*4, kernel)
#         self.conv2 = Conv(out_channels*4, out_channels*2, kernel)
#         self.down1 = Down(out_channels*2, out_channels, kernel)
        
#     def forward(self, x):
#         x = torch.swapaxes(x, 1, 2)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.down1(x)
#         x = torch.swapaxes(x, 1, 2)
#         return x

# class BehavEmbedding(pl.LightningModule):
#     def __init__(self, in_channels, out_channels, kernel):
#         super().__init__()
#         self.conv1 = Conv(in_channels, out_channels//2, kernel)
#         self.conv2 = Conv(out_channels//2, out_channels, kernel)
        
#     def forward(self, x):
#         x = torch.swapaxes(x, 1, 2)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = torch.swapaxes(x, 1, 2)
#         return x
# ############################################################


class ConvLayerNorm(pl.LightningModule):
        def __init__(self, in_channels, out_channels, kernel, length):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same", bias=True),
                nn.Dropout(constants.dropout),
                nn.LayerNorm([out_channels, length]),
                nn.LeakyReLU(0.2)
            )
        def forward(self, x1):
            return self.conv(x1)

              
class Conv(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same", bias=True),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1):
        return self.conv(x1)


class DoubleConv(pl.LightningModule):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel, padding="same", bias=False),
            nn.Dropout(constants.dropout),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class DoubleConvLayerNorm(pl.LightningModule):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel, length, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=kernel, padding="same", bias=True),
            nn.Dropout(constants.dropout),
            nn.LayerNorm([mid_channels, length]),
            nn.LeakyReLU(0.2),
            nn.Conv1d(mid_channels, out_channels, kernel_size=kernel, padding="same", bias=True),
            nn.Dropout(constants.dropout),
            nn.LayerNorm([out_channels, length]),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.double_conv(x)

    
class Down(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class DownLayerNorm(pl.LightningModule):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel, length):
        super().__init__()
        length = length // 2
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConvLayerNorm(in_channels, out_channels, kernel, length)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
    
class Up(pl.LightningModule):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel, in_channels)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=kernel, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,  kernel)

    def forward(self, x1, x2 = None):
        if(x2 == None):
            x1 = self.up(x1)
            return self.conv(x1)
        else:
            x1 = self.up(x1)
            diff = x2.size()[2] - x1.size()[2]
            x1 = F.pad(x1, [diff // 2, diff - diff// 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)


class OutConv(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding="same")

    def forward(self, x):
        return self.conv(x)