import torch

from typing import Tuple, List

def evaluateClassificationModel(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                getProbs: bool=False,
                                device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor|None]:
    model.eval()

    preds = []
    targets = []

    if getProbs:
        probs = []

    with torch.inference_mode():
        for batch in dataloader:
            xh, xa = batch["home"].to(device), batch["away"].to(device)
            mh, ma = batch["mask_home"].to(device), batch["mask_away"].to(device)

            ylogit = model(xh, xa, mh, ma)
            yprob = torch.softmax(ylogit, dim=1)
            ypred = torch.argmax(yprob, dim=1)

            if getProbs:
                probs.append(yprob.cpu())

            preds.append(ypred.cpu())
            targets.append(batch["y"].squeeze(-1).long())

    if getProbs:
        probs = torch.cat(probs)
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return preds, targets, probs if getProbs else None

def extractCalibrationSignals(preds: torch.Tensor,
                              targets: torch.Tensor,
                              probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    confidences, _ =probs.max(dim=1)
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