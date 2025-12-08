import torch

from typing import List, Dict
from torch import nn

actDict = {
    "SiLU": lambda ns, ipl: nn.SiLU(inplace=ipl),
    "ReLU": lambda ns, ipl: nn.ReLU(inplace=ipl),
    "LeakyReLU": lambda ns, ipl: nn.LeakyReLU(negative_slope=ns, inplace=ipl)
}

def getActivation(activation: str, inplace: bool=True, negativeSlope: float=0.01):
    assert activation in actDict, f"activation must be one of: {actDict.keys()}"
    return actDict[activation](negativeSlope, inplace)

class Residual(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
        return x + self.module(x, mask)

class ConvBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int=3, norm: bool=True, activation: str='SiLU'):
        assert inChannels > 0, "inChannels must be a positive integer"
        assert inChannels > 0, "outChannels must be a positive integer"
        assert kernelSize > 0, "inChannels must be a positive integer"
        super().__init__()
        self.block = nn.ModuleList()
        if norm:
            self.block.append(nn.GroupNorm(num_groups=1, num_channels=inChannels))
        self.block.append(nn.Conv1d(in_channels=inChannels, 
                                    out_channels=outChannels,
                                    kernel_size=kernelSize,
                                    padding=kernelSize >> 1))
        self.block.append(getActivation(activation=activation))
        self.block = nn.Sequential(*self.block)
    
    def forward(self, x: torch.Tensor, mask: None|torch.Tensor=None) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.block(x).transpose(1, 2)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, channels, numHeads: int=1, dropout: float=0.0):
        assert channels > 0, "channels must be a positive integer"
        assert numHeads > 0, "numHeads must be a positive integer"
        assert 0.0 <= dropout < 1.0, "dropout must be a float in range [0, 1)"
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=channels)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=numHeads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, mask: None|torch.Tensor=None) -> torch.Tensor:
        x = self.ln(x)
        return self.mha(
            query=x, key=x, value=x,
            key_padding_mask=mask
        )[0]

class TokenEmbedding(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], embDim: int):
        super().__init__()
        self.embDim = embDim

        self.tokenIndexes = sorted(vocabSizes.keys())
        self.embLayers = nn.ModuleList([
            nn.Embedding(
                num_embeddings=vocabSizes[i],
                embedding_dim=embDim,
                padding_idx=0
            )
            for i in self.tokenIndexes
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        out = x.new_empty((B, T, F, self.embDim)) 
        for i, emb in zip(self.tokenIndexes, self.embLayers):
            out[:, :, i, :] = emb(x[:, :, i].to(dtype=torch.int32))
        return out

class FeatureProjector(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], embDim: int, numFeatures: int=20, activation: str="SiLU"):
        super().__init__()
        self.embDim = embDim
        self.embedding = TokenEmbedding(vocabSizes=vocabSizes, embDim=embDim)

        tokenIndexes = set(self.embedding.tokenIndexes)
        self.contIndexes = [i for i in range(numFeatures) if i not in tokenIndexes]

        self.contProjs = nn.ModuleList([
            nn.Linear(1, embDim) for _ in self.contIndexes
        ])
        self.activation = getActivation(activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        
        for i, proj in zip(self.contIndexes, self.contProjs):
            out[:, :, i, :] = self.activation(proj(x[:, :, i].unsqueeze(-1)))
        
        return out
    
class DownsampleBlock(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, 
                 attnBlocks: int=1, numAttnHeads: int=1, attnDropout: float=0.0,
                 resAttn: bool=True, convBlocks: int=1, convKernelSize: int=3, 
                 convNorm: bool=True, convActivation: str="SiLU", resConv: bool=True):
        assert attnBlocks > 0 or convBlocks > 0, "downsample block must contain atleast one attention block or convolutional block"
        super().__init__()
        
        self.attns = nn.ModuleList()
        for _ in range(attnBlocks):
            self.attns.append(Residual(AttentionBlock(channels=inChannels, numHeads=numAttnHeads, dropout=attnDropout))
                             if resAttn else AttentionBlock(channels=inChannels, numHeads=numAttnHeads, dropout=attnDropout))
            
        self.convs = nn.ModuleList()
        for i in range(convBlocks):
            if resConv and i < convBlocks - 1:
                self.convs.append(Residual(ConvBlock(inChannels=inChannels, outChannels=inChannels, kernelSize=convKernelSize, 
                                                     norm=convNorm, activation=convActivation)))
                continue
            out = inChannels if i < convBlocks - 1 else outChannels
            self.convs.append(ConvBlock(inChannels=inChannels, outChannels=out, kernelSize=convKernelSize, 
                                        norm=convNorm, activation=convActivation))

    def forward(self, x: torch.Tensor, mask: None|torch.Tensor=None) -> torch.Tensor:
        for attn in self.attns:
            x = attn(x, mask)
        for conv in self.convs:
            x = conv(x, mask)
        return x

### Under Development ###
class Encoder(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], embDim: int, numFeatures: int=20):
        super().__init__()
        # embed tokens, shape to z
        # convolution, attention and residual
        self.featProject = FeatureProjector(vocabSizes=vocabSizes, embDim=embDim, numFeatures=numFeatures)
        

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.Tensor()
    
### Under Development ###
class MLP(nn.Module):
    def __init__(self,):
        super().__init__()
        # x -> logit of assigned shape
        # convolution, linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor()
    
### Under Development ###
class FeatureExtractor(nn.Module):
    def __init__(self,):
        super().__init__()
        # extract features (mid block of nn architecture)
        # attention, residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.Tensor()

### Under Development ###
class MatchPredictorV0(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], seqLen: int=20, numFeatures: int=20, embDim: int=1):
        assert seqLen > 0, "seqLen must be a positive integer"
        assert numFeatures > 0, "numFeatures must be a postive integer"
        assert embDim > 0, "embDim must be a positive integer"
        super().__init__()
        # pass xh, xa through the Encoder seperately
        # concatenate both Encoder outputs
        # feature extraction of new concetanated tensor
        # mlp to get logit (y)
        self.encoder = Encoder(vocabSizes=vocabSizes, embDim=embDim, numFeatures=numFeatures)
        self.feature = FeatureExtractor()
        self.mlp = MLP()

    def forward(self, xh: torch.Tensor, xa: torch.Tensor, mh: torch.Tensor, ma: torch.Tensor) -> torch.Tensor:
        xh, xa = self.encoder(xh, mh), self.encoder(xa, ma)
        x = torch.cat([xh, xa], dim=-1)
        x += self.feature(x)
        return self.mlp(x)