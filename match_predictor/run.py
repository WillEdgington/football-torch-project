import torch

from torchinfo import summary
from tensor_pipeline import prepareData

from .models import DownsampleBlock, FeatureProjector, Encoder, FeatureExtractor, MLP, MatchPredictorV0

if __name__=="__main__":
    dataloader = prepareData(type="train")
    if not (dataloader is None or isinstance(dataloader, dict)):
        batch = next(iter(dataloader)) if dataloader is not None else {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape, v.dtype)
                continue
            # print(k, len(v), len(v[0]))

        tokenCols = dataloader.dataset.tokenCols
        vocabSize = {
            i: s for i, s in zip(tokenCols["index"], tokenCols["size"])
        }
        print(vocabSize)
        fp = FeatureProjector(vocabSizes=vocabSize, embDim=2, numFeatures=60)
        summary(fp, input_size=(64, 20, 60))

        db = DownsampleBlock(inChannels=120, outChannels=30, attnBlocks=3, convBlocks=3)
        summary(db, input_size=[(64, 20, 120), (64, 20)])

        enc = Encoder(vocabSizes=vocabSize, embDim=2, numFeatures=60, outChannels=10, numDownBlocks=5, attnBlocksPerDown=4)
        summary(enc, input_size=[(64, 20, 60), (64, 20)])

        fe = FeatureExtractor(numFeatures=20, depth=5)
        summary(fe, input_size=[(64, 40, 20), (64, 40)])

        mlp = MLP(channels=40, numFeatures=20, outDim=3)
        summary(mlp, input_size=(64, 40, 20))

        model = MatchPredictorV0(vocabSizes=vocabSize, outDim=3, seqLen=20, 
                                 embDim=2, numFeatures=60, latentSize=20, 
                                 encoderNumDownBlocks=5, encoderAttnBlocksPerDown=4,
                                 featExtractorDepth=5)
        summary(model, input_size=[(64, 20, 60), (64, 20, 60), (64, 20), (64, 20)])