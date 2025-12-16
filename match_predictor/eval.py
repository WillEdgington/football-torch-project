import torch

from typing import Tuple

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