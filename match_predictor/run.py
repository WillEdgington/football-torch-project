import torch
import random

from torchinfo import summary
from tensor_pipeline import prepareData, RandomTokenUNK, TemporalDropout, MissingValueAugment, ContinuousFeatureDropout, Compose

from .models import MatchPredictorV0
from .train import train
from utils.save import saveStates, saveTorchObject, loadResultsMap, loadStates
from .config import SAVEDMODELSDIR, BIGFIVELEAGUES
from .plots import plotLoss, plotAccuracy, plotResults, plotConfusionMatrix, plotClassConfidenceHistogram, plotReliabilityDiagram

device = "cuda" if torch.cuda.is_available() else "cpu"
MANUALSEED = 42
torch.manual_seed(seed=MANUALSEED)
random.seed(MANUALSEED)

BATCHSIZE = 128
LR = 0.0001 * (BATCHSIZE / 64)

SEQLEN = 50
LATENTSIZE = 20
EMBDIM = 1
ENCODERDEPTH = 2
FEATEXTDEPTH = 2

EPOCHS = 20
SAVEPOINT = 10

TRIALDIR = SAVEDMODELSDIR + "/TRIAL_11_SHARDING_STORE"
MODELNAME = f"MODEL_RTU0709_TD0101_MVA01005_CFD01005_BS{BATCHSIZE}_SL{SEQLEN}_LS{LATENTSIZE}_EMBD{EMBDIM}_EABPD1_ENDB{ENCODERDEPTH}_EAD03_EAH2_FED{FEATEXTDEPTH}_FEAD03_FEAH2"
RESULTSNAME = f"{MODELNAME}_RESULTS.pt"

if __name__=="__main__":
    augmentation = Compose([
                        RandomTokenUNK(prob=0.7, intensity=0.9),
                        TemporalDropout(prob=0.1, minKeep=1),
                        MissingValueAugment(prob=0.1, intensity=0.05),
                        ContinuousFeatureDropout(prob=0.1, intensity=0.05)
    ])

    dataloaders = prepareData(batchSize=BATCHSIZE, trainTransform=augmentation, seqLen=SEQLEN)
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
        # print(f"{v[0]}\n")
    model = MatchPredictorV0(vocabSizes=vocabSize, outDim=3, seqLen=SEQLEN,
                             embDim=EMBDIM, numFeatures=60, latentSize=LATENTSIZE,
                             encoderNumDownBlocks=ENCODERDEPTH, encoderAttnBlocksPerDown=1,
                             featExtractorDepth=FEATEXTDEPTH, encoderAttnDropout=0.3, featExtractorAttnDropout=0.3,
                             encoderNumAttnHeads=2, featExtractorNumAttnHeads=2)
    model = model.to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=0.0001)

    if epochsComplete > 0:
        states = loadStates(stateName=f"{MODELNAME}_EPOCHS_{epochsComplete}.pt",
                            stateDir=TRIALDIR,
                            model=model,
                            optimizer=optimizer)
    model = model.to(device)
    summary(model, 
            input_size=[
                (64, SEQLEN, 60),
                (64, SEQLEN, 60),
                (64, SEQLEN),
                (64, SEQLEN),
                (64, SEQLEN, 49),
                (64, SEQLEN, 49)
            ])

    lossFn = torch.nn.CrossEntropyLoss()
    
    while epochsComplete < EPOCHS:
        results = train(model=model,
                        trainDataloader=trainDataloader,
                        testDataloader=valDataloader,
                        lossFn=lossFn,
                        optimizer=optimizer,
                        epochs=SAVEPOINT,
                        seed=MANUALSEED,
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

    bigFiveLoaders = prepareData(batchSize=BATCHSIZE, seqLen=SEQLEN, groups=BIGFIVELEAGUES)
    plotConfusionMatrix(model=model, dataloader=bigFiveLoaders["test"], title=f"{MODELNAME} (test) -")
    plotClassConfidenceHistogram(model=model, dataloader=bigFiveLoaders["test"], title=f"{MODELNAME} (test) -")
    plotClassConfidenceHistogram(model=model, dataloader=bigFiveLoaders["train"], title=f"{MODELNAME} (train) -")
    plotReliabilityDiagram(model=model, dataloader=bigFiveLoaders["test"], bins=20, title=f"{MODELNAME} (test) -")