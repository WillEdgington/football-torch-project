import torch

from typing import Tuple, List, Dict, Any

def evaluateClassificationModel(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                getProbs: bool=False,
                                getLoss: bool=False,
                                device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None, float|None]:
    model.eval()

    preds = []
    targets = []

    if getProbs:
        probs = []
    
    if getLoss:
        evalLoss = 0
        lossFn = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for batch in dataloader:
            xh, xa = batch["home"].to(device), batch["away"].to(device)
            mh, ma = batch["mask_home"].to(device), batch["mask_away"].to(device)
            missh = batch["missing_home"].to(device) if "missing_home" in batch else None
            missa = batch["missing_away"].to(device) if "missing_away" in batch else None
            Y = batch["y"].to(device).squeeze(-1).long()

            with torch.amp.autocast(device_type=device, enabled=(device=="cuda")):
                ylogit = model(xh, xa, mh, ma, missh, missa)
                if getLoss: loss = lossFn(ylogit, Y)
            if getLoss: evalLoss += loss.item()
            yprob = torch.softmax(ylogit, dim=1)
            ypred = torch.argmax(yprob, dim=1)

            if getProbs:
                probs.append(yprob.cpu())

            preds.append(ypred.cpu())
            targets.append(Y)

    if getProbs:
        probs = torch.cat(probs)
    
    if getLoss:
        evalLoss /= len(dataloader)
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return preds, targets, probs if getProbs else None, evalLoss if getLoss else None

def extractCalibrationSignals(preds: torch.Tensor,
                              targets: torch.Tensor,
                              probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    confidences, _ = probs.max(dim=1)
    correct = (preds == targets).float()
    return confidences, correct

def binCalibration(confidences: torch.Tensor,
                   correct: torch.Tensor,
                   bins: int=10) -> Tuple[List[float], List[float], List[float], List[int]]:
    binEdges = torch.linspace(0.0, 1.0, bins + 1)
    binCenters = 0.5 * (binEdges[:-1] + binEdges[1:])

    binAcc = []
    binConf = []
    binCount = []

    for i in range(bins):
        mask = (confidences >= binEdges[i]) & (confidences < binEdges[i + 1])

        if mask.any():
            binAcc.append(correct[mask].mean().item())
            binConf.append(confidences[mask].mean().item())
            binCount.append(mask.sum().item())
        else:
            binAcc.append(0.0)
            binConf.append(0.0)
            binCount.append(0)
        
    return binCenters, binAcc, binConf, binCount

def computeECE(binAcc: List[float], 
               binConf: List[float], 
               binCount: List[int]) -> float:
    binCount = torch.tensor(binCount, dtype=torch.float)
    total = binCount.sum()

    ece = 0.0
    for acc, conf, cnt in zip(binAcc, binConf, binCount):
        if cnt > 0:
            ece += (cnt / total) * abs(acc - conf)

    return ece.item()

def computeConfusionMatrix(numClasses: int,
                           preds: torch.Tensor,
                           targets: torch.Tensor) -> torch.Tensor:
    conf = torch.zeros((numClasses, numClasses), dtype=torch.int64)
    for t, p in zip(targets, preds):
        conf[t, p] += 1
    return conf

def evaluateMatchPredictorModel(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                bins: int=20,
                                device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, Any]:
    preds, targets, probs, evalLoss = evaluateClassificationModel(model=model,
                                                                  dataloader=dataloader,
                                                                  getProbs=True,
                                                                  getLoss=True,
                                                                  device=device)
    
    accuracy = (preds == targets).float().mean().item()

    confMat = computeConfusionMatrix(numClasses=probs.shape[-1],
                                     preds=preds,
                                     targets=targets)

    confidences, correct = extractCalibrationSignals(preds=preds, 
                                                     targets=targets,
                                                     probs=probs)

    binCenters, binAcc, binConf, binCount = binCalibration(confidences=confidences,
                                                           correct=correct,
                                                           bins=bins)

    ece = computeECE(binAcc=binAcc,
                     binConf=binConf,
                     binCount=binCount)
    
    return {
        "loss": float(evalLoss),
        "accuracy": accuracy,
        "ece": float(ece),
        "confusion_matrix": confMat.numpy(),
        "calibration": {
            "bin_centers": binCenters,
            "bin_accuracy": binAcc,
            "bin_confidence": binConf,
            "bin_count": binCount,
        }
    }


# EvaluatorFn = Callable[[Trial], Dict[str, Any]]
# we could input an evaluator function in Experiment.__init__()?
# "evals": {
#     test data label (e.g. "big five leagues testDataloader"): {
#         "loss": loss,
#         "accuracy": accuracy,
#         "confusion matrix": matrix,
#         "ECE": ECE,
#         other evaluations to bundle
#     }            
# }