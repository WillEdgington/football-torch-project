import torch

from torchinfo import summary
from tensor_pipeline import prepareData, RandomTokenUNK

from .models import MatchPredictorV0
from .train import train
from .save import saveStates, saveTorchObject, loadResultsMap, loadStates
from .config import SAVEDMODELSDIR
from .plots import plotLoss, plotAccuracy, plotResults, plotConfusionMatrix, plotClassConfidenceHistogram, plotReliabilityDiagram

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
torch.manual_seed(seed=MANUALSEED)

BATCHSIZE = 64
LR = 0.0001 * (BATCHSIZE / 64)

EPOCHS = 100
SAVEPOINT = 10

TRIALDIR = SAVEDMODELSDIR + "/TRIAL_1_RANDOMTOKENUNK"
MODELNAME = "MODEL_0"
RESULTSNAME = f"{MODELNAME}_RESULTS.pt"

if __name__=="__main__":
    augmentation = RandomTokenUNK(prob=0.8, intensity=0.4)

    dataloaders = prepareData(batchSize=BATCHSIZE, trainTransform=augmentation)
    assert isinstance(dataloaders, dict), "dataloaders is not type dict"
    trainDataloader, valDataloader, testDataloader = dataloaders["train"], dataloaders["validation"], dataloaders["test"]

    results = loadResultsMap(resultsDir=TRIALDIR, resultsName=RESULTSNAME)
    epochsComplete = min(len(results["train_loss"]), EPOCHS) if results is not None else 0
    
    tokenCols = trainDataloader.dataset.tokenCols
    vocabSize = {
        i: s for i, s in zip(tokenCols["index"], tokenCols["size"])
    }
    batch = next(iter(trainDataloader))
    for k, v in batch.items():
        print(f"{k}, {v.shape}")
    model = MatchPredictorV0(vocabSizes=vocabSize, outDim=3, seqLen=20, 
                             embDim=1, numFeatures=60, latentSize=20,
                             encoderNumDownBlocks=1, encoderAttnBlocksPerDown=1,
                             featExtractorDepth=1, encoderAttnDropout=0.3, featExtractorAttnDropout=0.3)
    model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=0.0001)

    if epochsComplete > 0:
        states = loadStates(stateName=f"{MODELNAME}_EPOCHS_{epochsComplete}.pt",
                            stateDir=TRIALDIR,
                            model=model, 
                            optimizer=optimizer)
    model.to(device)
    summary(model, input_size=[(64, 20, 60), (64, 20, 60), (64, 20), (64, 20)])

    lossFn = torch.nn.CrossEntropyLoss()
    
    while epochsComplete < EPOCHS:
        results = train(model=model,
                        trainDataloader=trainDataloader,
                        testDataloader=valDataloader,
                        lossFn=lossFn,
                        optimizer=optimizer,
                        epochs=SAVEPOINT,
                        results=results,
                        calcAccuracy=True,
                        enableAmp=True,
                        gradClipping=1.0,
                        device=device)
        epochsComplete += SAVEPOINT
        saveTorchObject(obj=results, targetDir=TRIALDIR, fileName=RESULTSNAME)
        saveStates(stateName=f"{MODELNAME}_EPOCHS_{epochsComplete}.pt", 
                   stateDir=TRIALDIR,
                   model=model,
                   optimizer=optimizer)
    plotResults(results=results, title=f"{MODELNAME} results")
    plotConfusionMatrix(model=model, dataloader=testDataloader, title=f"{MODELNAME} (test) -")
    plotClassConfidenceHistogram(model=model, dataloader=testDataloader, title=f"{MODELNAME} (test) -")
    plotClassConfidenceHistogram(model=model, dataloader=trainDataloader, title=f"{MODELNAME} (train) -")
    plotReliabilityDiagram(model=model, dataloader=testDataloader, bins=20, title=f"{MODELNAME} (test) -")