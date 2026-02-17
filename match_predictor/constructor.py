import torch

from typing import Dict, Tuple, Any

from experiments import Trial
from tensor_pipeline.transforms import RandomTokenUNK, TemporalDropout, MissingValueAugment, ContinuousFeatureDropout, Compose, Transform
from tensor_pipeline import prepareData
from utils.save import loadStates

from .models import MatchPredictorV0

def constructTransform(name: str, params: Dict[str, Any]) -> Transform:
    match name:
        case "RandomTokenUNK":
            return RandomTokenUNK(prob=params["prob"],
                                  intensity=params["intensity"])
        case "TemporalDropout":
            return TemporalDropout(prob=params["prob"],
                                   minKeep=params["minKeep"])
        case "MissingValueAugment":
            return MissingValueAugment(prob=params["prob"],
                                       intensity=params["intensity"])
        case "ContinuousFeatureDropout":
            return ContinuousFeatureDropout(prob=params["prob"],
                                            intensity=params["intensity"])
        case _:
            raise RuntimeError(f'''{name} is not an elligable transform.
Elligable Transforms:
    - RandomTokenUNK
    - TemporalDropout
    - MissingValueAugment
    - ContinuousFeatureDropout''')

def prepareMatchPredictorTransform(params: Dict[str, Any]|None) -> Transform|None:
    if params is None:
        return None
    transforms = [constructTransform(name=key, params=val) for key, val in params.items()]
    if len(transforms) > 1:
        return Compose(transforms)
    return transforms[0]

# definition["data"]["transform"]: (
#   None |
#   {
#       "{transform name}": {
#           "prob": prob,
#           "intensity": intensity,
#           you get the idea
#       } 
#   }
# if None then augmentation = None else pass into prepareMatchPredictorTransform

def prepareMatchPredictorData(params: Dict[str, Any],
                              seed: int) -> Dict[str, torch.utils.data.DataLoader]:
    augmentation = prepareMatchPredictorTransform(params=params.get("transform", None))
    dataloaders = prepareData(batchSize=params.get("batchSize", 64),
                              seqLen=params.get("seqLen", 20),
                              trainTransform=augmentation,
                              groups=params.get("groups", None),
                              seed=seed)
    return dataloaders

# definition["data"] = {
#     "transform": transform info,
#     "batchSize": batch size,
#     "seqLen": sequence length,
#     "groups": groups for train data (eval is seperate, groupCol="league")
# }

def constructMatchPredictorModel(params: Dict[str, Any],
                                 dataloader: Dict[str, torch.utils.data.DataLoader]|torch.utils.data.DataLoader,
                                 device: torch.device) -> torch.nn.Module:
    assert params["model"] == "MatchPredictorV0", f'{params["model"]} is not an elligable model for constructor.\nElligable Models:\n  - MatchPredictorV0'
    
    if isinstance(dataloader, dict):
        dataloader = list(dataloader.values())[0]
    tokenCols = dataloader.dataset.tokenCols
    vocabSize = {
        i: s for i, s in zip(tokenCols["index"], tokenCols["size"])
    }
    model = MatchPredictorV0(vocabSizes=vocabSize,
                             seqLen=dataloader.dataset.seqLen,
                             embDim=params.get("embDim", 2),
                             latentSize=params.get("latentSize", 10),
                             encoderNumDownBlocks=params.get("encoderNumDownBlocks", 1),
                             encoderAttnBlocksPerDown=params.get("encoderAttnBlocksPerDown", 1),
                             encoderNumAttnHeads=params.get("encoderNumAttnHeads", 1),
                             encoderAttnDropout=params.get("encoderAttnDropout", 0.0),
                             encoderResAttn=params.get("encoderResAttn", "true")=="true",
                             encoderConvBlocksPerDown=params.get("encoderConvBlocksPerDown", 1),
                             encoderConvKernelSize=params.get("encoderConvKernelSize", 3),
                             encoderConvNorm=params.get("encoderConvNorm", "true")=="true",
                             encoderConvActivation=params.get("encoderConvActivation", "SiLU"),
                             encoderResConv=params.get("encoderResConv", "true")=="true",
                             featExtractorDepth=params.get("featExtractorDepth", 2),
                             featExtractorUseAttn=params.get("featExtractorUseAttn", "true")=="true",
                             featExtractorResAttn=params.get("featExtractorResAttn", "true")=="true",
                             featExtractorAttnDropout=params.get("featExtractorAttnDropout", 0.0),
                             featExtractorNumAttnHeads=params.get("featExtractorNumAttnHeads", 2),
                             featExtractorUseFFN=params.get("featExtractorUseFFN", "true")=="true",
                             featExtractorResFFN=params.get("featExtractorResFFN", "true")=="true",
                             featExtractorExpansionFFN=params.get("featExtractorExpansionFFN", 2),
                             featExtractorLnormFFN=params.get("featExtractorLnormFFN", "true")=="true",
                             featExtractorActivationFFN=params.get("featExtractorActivationFFN", "SiLU"),
                             activationMLP=params.get("activationMLP", "SiLU")).to(device=device)
    return model

def constructOptimizer(params: Dict[str, Any],
                       model: torch.nn.Module) -> torch.optim.Optimizer:
    name = params["optimizer"]
    match name:
        case "AdamW":
            return torch.optim.AdamW(params=model.parameters(), 
                                     lr=params.get("lr", 0.001), 
                                     weight_decay=params.get("weight_decay", 0.01))
        case "Adam":
            return torch.optim.Adam(params=model.parameters(), 
                                    lr=params.get("lr", 0.001))
        case "SGD":
            return torch.optim.SGD(params=model.parameters(),
                                   lr=params.get("lr", 0.001),
                                   momentum=params.get("momentum", 0))
        case _:
            raise RuntimeError(f'''{name} is not an elligable optimizer.
Elligable Optimizers:
    - AdamW
    - Adam
    - SGD''')
        
def prepareLossFunction(params: Dict[str, Any]) -> torch.nn.Module:
    name = params.get("lossFn", "CrossEntropyLoss")
    match name:
        case "CrossEntropyLoss":
            return torch.nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0))
        case _:
            raise RuntimeError(f'''{name} is not an elligable loss function.
Elligable Loss Functions:
    - CrossEntropyLoss''')

def prepareMatchPredictorState(definition: Dict[str, Any],
                               dataloader: Dict[str, torch.utils.data.DataLoader]|torch.utils.data.DataLoader,
                               state: Dict[str, Any],
                               modelDir: str,
                               train: bool=True) -> Dict[str, Any]:
    device = definition.get("device", "cpu")
    model = constructMatchPredictorModel(params=definition["model"],
                                         dataloader=dataloader,
                                         device=device)
    optimizer = constructOptimizer(params=definition["optimizer"], model=model) if train else None
    
    epochsCompleted = state["epochs_completed"]
    if epochsCompleted > 0:
        loadStates(stateName=f"MODEL_{epochsCompleted}_EPOCHS.pt",
                   stateDir=modelDir,
                   model=model,
                   optimizer=optimizer)

    return {
        "model": model.to(device=device),
        "lossFn": prepareLossFunction(params=definition["lossFn"]) if train else None,
        "optimizer": optimizer
    }

def constructMatchPredictorTrial(trial: Trial,
                                 train: bool=True) -> Dict[str, 
                                                       torch.nn.Module|
                                                       torch.optim.Optimizer|
                                                       Dict[str,torch.utils.data.DataLoader]]:
    definition = trial.getDefinition()
    state = trial.getState()
    
    seed = definition.get("seed", 42)
    dataParams = definition["data"]
    dataloaders = prepareMatchPredictorData(params=dataParams,
                                            seed=seed)
    loadedState = prepareMatchPredictorState(definition=definition,
                                             dataloader=dataloaders,
                                             state=state,
                                             modelDir=trial.modelPath)
    loadedState["dataloaders"] = dataloaders
    return loadedState

# definition = {
#     "model": {
#         "model": "MatchPredictorV0",
#     },
#     "optimizer": {
#         "optimizer": "AdamW",
#         "lr": 0.0001
#     },
#     "data": {
#         "transform": None,
#         "batchSize": 32,
#     },
#     "lossFn": {
#         "lossFn": "CrossEntropyLoss"
#     },
#     "device": "cpu",
#     "seed": 42
# }

# trial = Trial.create(definition=definition)
# print(constructMatchPredictorTrial(trial))