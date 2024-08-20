import math
import torch
import torch.nn as nn

from preprocess import PREFIX_TO_TRAFFIC_ID, PREFIX_TO_APP_ID, AUX_ID
from train import train_op
import pdb
from math import ceil
from mtt_block import MTT_Block

class CustomEmbedding(nn.Module):
    """Embedding layer"""

    def __init__(self, n_channels: int, n_dims: int):
        """
        Args:
            n_channels: Channels of embedding.
            n_dims: The dimensions of embedding.
        """
        super(CustomEmbedding, self).__init__()
        self.n_channels = n_channels
        self.n_dims = n_dims
        self.embedding = nn.Linear(self.n_dims, self.n_dims)

    def forward(self, input_data):
        """

        Args:
            input_data: Intput data with shape(B, C, L)

        Returns:
            Data with shape(B, Channel, Dimension)
        """
        input_data = input_data[:, :, 0:1440]
        input_data = input_data.reshape(-1, self.n_channels, self.n_dims)
        embedded = self.embedding(input_data)

        return embedded


class PositionalEncoding(nn.Module):
    """Position encoding layer in transformer model"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # ps shape ``[seq_len, batch_size, embedding_dim]``

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        pe = self.pe[:x.size(1), :].permute(1, 0, 2)
        # x = x + pe
        x = torch.add(x, pe)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Normal transformer block"""

    def __init__(
            self,
            embed_dim: int,
            n_heads: int = 600,
            d1: int = 768,
            d2: int = 5000,
            dropout: float = 0.1
    ):
        """Initialize the transformer block.

        Args:
            embed_dim: Embedding dimensions from the output of embedding layer.
            n_heads: Number of heads for multi-head attention.
            d1: The output dimensions of the first fully connected layer.
            d2: The output dimensions of the second fully connected layer.
            dropout: Dropout ratio.
        """
        super(TransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads=n_heads,
                                                    batch_first=True)
        # (N,L,E) when batch_first=True
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, d1),
            nn.ReLU(),
            nn.Linear(d1, d2),
            nn.Dropout(dropout)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.embed_size = embed_dim

    def forward(self, x, mask=None):
        # Multihead attention
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        attn_output = self.dropout1(attn_output)
        # Residual connection and layer normalization
        x = self.layer_norm1(x + attn_output)

        # Feedforward network
        ff_output = self.feedforward(x)
        ff_output = self.dropout2(ff_output)
        # Residual connection and layer normalization
        x = self.layer_norm2(x + ff_output)

        return x





# new adding
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# new adding
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# new adding
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x






# new adding
class TokenDropAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        print(num_heads)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # pdb.set_trace()
        left_tokens = N - 1
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens

        return  x, None, None, None, left_tokens


# new adding
class TokenDropBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_dim=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = TokenDropAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate
        B, N, C = x.shape

        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        if index is not None:
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            # pdb.set_trace()
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            # if self.fuse_token:
            #     compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
            #     non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C]

            #     non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens]
            #     extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
            #     x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            # else:
            x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 1
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None



class MTT_dropping(nn.Module):
    def __init__(
            self,
            seq_len=1440,
            embed_n=90,
            embed_d=16,
            trans_h=4,
            trans_d1=256,
            n_block=2,
            keep_rate=1.0
    ):
        super(MTT_dropping, self).__init__()
        self.seq_len = seq_len
        self.embed_n = embed_n
        self.embed_d = embed_d
        self.n_block = n_block

        self.embedding = CustomEmbedding(embed_n, embed_d)
        self.pos_encoder = PositionalEncoding(embed_d, dropout=0.1)

        
        # original code below
        # encoder_layer = nn.TransformerEncoderLayer(d_model=embed_d, nhead=trans_h, dim_feedforward=trans_d1,
        #                                            batch_first=True)
        # self.transformer_blk1 = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.drop_blk1 = TokenDropBlock(
            dim=16,
            num_heads=1, 
            mlp_dim=trans_d1,
            keep_rate=keep_rate)

        if (n_block > 1):
            self.mtt_block = MTT_Block(
                n_block=n_block-1,
                trans_h=trans_h,
                trans_d1=trans_d1,
                embed_d=embed_d
            )
        

        # Task-specific layers
        num_label = 6
        if(keep_rate == 1):
            self.task1_output = nn.Linear(90*16, num_label)
        else:
            self.task1_output = nn.Linear(ceil(keep_rate*89)*16+16, num_label)


    def forward(self, x):
        x = self.embedding(x)  # Shape(B, embed_n*embed_d)
        x = self.pos_encoder(x)
        
        x = self.drop_blk1(x)[0]
        if (self.n_block > 1):
            x = self.mtt_block(x)
        
        x = torch.flatten(x, start_dim=1)
        
        # pdb.set_trace()

        output1 = self.task1_output(x)
        return output1


def train():
    model = MTT_dropping(n_block=2, keep_rate=0.8)
    task_weights = (4, 2, 1)
    train_op(model, task_weights=task_weights, n_epochs=10)


if __name__ == '__main__':
    train()
