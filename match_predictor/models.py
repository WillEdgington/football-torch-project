import torch

from typing import List, Dict
from torch import nn

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
    def __init__(self, vocabSizes: Dict[int, int], embDim: int, numFeatures: int=20):
        super().__init__()
        self.embDim = embDim
        self.embedding = TokenEmbedding(vocabSizes=vocabSizes, embDim=embDim)

        tokenIndexes = set(self.embedding.tokenIndexes)
        self.contIndexes = [i for i in range(numFeatures) if i not in tokenIndexes]

        self.contProjs = nn.ModuleList([
            nn.Linear(1, embDim) for _ in self.contIndexes
        ])
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.embedding(x)
        
        for i, proj in zip(self.contIndexes, self.contProjs):
            out[:, :, i, :] = self.activation(proj(x[:, :, i].unsqueeze(-1)))
        
        return out

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