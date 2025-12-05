import torch

from torchinfo import summary
from tensor_pipeline import prepareData

from .models import FeatureProjector

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
    