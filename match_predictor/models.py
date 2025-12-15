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
            keep = mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
            x = x * keep
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, channels: int, numHeads: int=1, dropout: float=0.0):
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
    
class FeedForwardBlock(nn.Module):
    def __init__(self, channels: int, expansion: int=2, activation: str="SiLU", lnorm: bool=True):
        assert expansion > 0, "expansion must be a positive integer"
        assert channels > 0, "channels must be a positive integer"
        super().__init__()
        layers = nn.ModuleList()
        if lnorm:
            layers.append(nn.LayerNorm(channels))
        layers += [
            nn.Linear(in_features=channels, out_features=channels * expansion),
            getActivation(activation=activation),
            nn.Linear(in_features=channels * expansion, out_features=channels)
        ]
        self.ffn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
        return self.ffn(x)
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], embDim: int):
        assert embDim > 0, "embDim must be a positive integer"
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
            out[:, :, i, :] = emb(x[:, :, i].long())
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
        attnmask = mask == 0 if mask is not None else None
        for attn in self.attns:
            x = attn(x, mask=attnmask)

        for conv in self.convs:
            x = conv(x, mask=mask)
        return x

class Encoder(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], embDim: int=2, 
                 numFeatures: int=60, outChannels: int=10, numDownBlocks: int=1,
                 attnBlocksPerDown: int=1, numAttnHeads: int=1, attnDropout: float=0.0,
                 resAttn: bool=True, convBlocksPerDown: int=1, convKernelSize: int=3, 
                 convNorm: bool=True, convActivation: str="SiLU", resConv: bool=True):
        assert outChannels > 0, "outChannels must be a positive integer"
        assert numDownBlocks > 0, "numDownBlocks must be a positive integer"
        super().__init__()
        self.featProject = FeatureProjector(vocabSizes=vocabSizes, embDim=embDim, numFeatures=numFeatures)
        inChannels = numFeatures * embDim
        channels = [inChannels + (((outChannels - inChannels) * i) // numDownBlocks) for i in range(numDownBlocks + 1)]
        self.downsamples = nn.ModuleList()
        for i in range(numDownBlocks):
            self.downsamples.append(
                DownsampleBlock(inChannels=channels[i], outChannels=channels[i+1], attnBlocks=attnBlocksPerDown, numAttnHeads=numAttnHeads,
                                attnDropout=attnDropout, resAttn=resAttn, convBlocks=convBlocksPerDown, convKernelSize=convKernelSize,
                                convNorm=convNorm, convActivation=convActivation, resConv=resConv)
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
        x = self.featProject(x)
        x = x.flatten(start_dim=2)
        for down in self.downsamples:
            x = down(x, mask)
        return x
    
class MLP(nn.Module):
    def __init__(self, channels: int, numFeatures: int, outDim: int, activation: str|None="SiLU"):
        assert channels > 0, "channels must be a positive integer"
        assert numFeatures > 0, "numFeatures must be a positive integer"
        assert outDim > 0, "outDim must be a positive integer"
        super().__init__()
        layers = nn.ModuleList()
        if activation is not None:
            layers.append(getActivation(activation=activation))
        layers.append(nn.Linear(channels * numFeatures, outDim))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        return self.head(x)
    
class FeatureExtractor(nn.Module):
    def __init__(self, numFeatures: int, depth: int=2, useAttn: bool=True,
                 resAttn: bool=True, attnDropout: float=0.0, numAttnHeads: int=2,
                 useFFN: bool=True, resFFN: bool=True, expansionFFN: int=2,
                 lnormFFN: bool=True, activationFFN: str="SiLU"):
        assert useAttn or useFFN, "useAttn or useFFN must be True"
        assert depth > 0, "depth must be a positive integer"
        assert numFeatures > 0, "channels must be a positive integer"
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            if useAttn:
                self.blocks.append(
                    Residual(AttentionBlock(channels=numFeatures, numHeads=numAttnHeads, dropout=attnDropout))
                    if resAttn else AttentionBlock(channels=numFeatures, numHeads=numAttnHeads, dropout=attnDropout)
                )
            
            if useFFN:
                self.blocks.append(
                    Residual(FeedForwardBlock(channels=numFeatures, expansion=expansionFFN, 
                                              activation=activationFFN, lnorm=lnormFFN))
                    if resFFN else FeedForwardBlock(channels=numFeatures, expansion=expansionFFN,
                                                    activation=activationFFN, lnorm=lnormFFN)
                )
        
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor|None=None) -> torch.Tensor:
        mask = mask == 0 if mask is not None else None
        for block in self.blocks:
            x = block(x, mask)
        return x

class MatchPredictorV0(nn.Module):
    def __init__(self, vocabSizes: Dict[int, int], outDim: int, seqLen: int=20, 
                 numFeatures: int=60, latentSize: int=10, embDim: int=2,
                 encoderNumDownBlocks: int=1, encoderAttnBlocksPerDown: int=1, encoderNumAttnHeads: int=1,
                 encoderAttnDropout: float=0.0, encoderResAttn: bool=True, encoderConvBlocksPerDown: int=1,
                 encoderConvKernelSize: int=3, encoderConvNorm: bool=True, encoderConvActivation: str="SiLU",
                 encoderResConv: bool=True, featExtractorDepth: int=2, featExtractorUseAttn: bool=True,
                 featExtractorResAttn: bool=True, featExtractorAttnDropout: float=0.0, featExtractorNumAttnHeads: int=2,
                 featExtractorUseFFN: bool=True, featExtractorResFFN: bool=True, featExtractorExpansionFFN: int=2,
                 featExtractorLnormFFN: bool=True, featExtractorActivationFFN: str="SiLU", activationMLP: str|None="SiLU"):
        assert latentSize > 0, "latentSize must be a positive integer"
        assert seqLen > 0, "seqLen must be a positive integer"
        assert numFeatures > 0, "numFeatures must be a postive integer"
        assert embDim > 0, "embDim must be a positive integer"
        super().__init__()
        self.encoder = Encoder(vocabSizes=vocabSizes, embDim=embDim, numFeatures=numFeatures,
                               outChannels=latentSize, numDownBlocks=encoderNumDownBlocks, attnBlocksPerDown=encoderAttnBlocksPerDown,
                               numAttnHeads=encoderNumAttnHeads, attnDropout=encoderAttnDropout, resAttn=encoderResAttn,
                               convBlocksPerDown=encoderConvBlocksPerDown, convKernelSize=encoderConvKernelSize, convNorm=encoderConvNorm,
                               convActivation=encoderConvActivation, resConv=encoderResConv)

        self.feature = FeatureExtractor(numFeatures=latentSize, depth=featExtractorDepth, useAttn=featExtractorUseAttn, 
                                        resAttn=featExtractorResAttn, attnDropout=featExtractorAttnDropout, numAttnHeads=featExtractorNumAttnHeads,
                                        useFFN=featExtractorUseFFN, resFFN=featExtractorResFFN, expansionFFN=featExtractorExpansionFFN,
                                        lnormFFN=featExtractorLnormFFN, activationFFN=featExtractorActivationFFN)
        
        self.mlp = MLP(channels=seqLen*2, numFeatures=latentSize, outDim=outDim)

    def forward(self, xh: torch.Tensor, xa: torch.Tensor, mh: torch.Tensor, ma: torch.Tensor) -> torch.Tensor:
        xh, xa = self.encoder(xh, mh), self.encoder(xa, ma)
        x = torch.cat([xh, xa], dim=1)
        m = torch.cat([mh, ma], dim=1)
        x = self.feature(x, m)
        return self.mlp(x)