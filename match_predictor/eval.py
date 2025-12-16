import torch

from typing import Tuple

def evaluateClassificationModel(model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    preds = []
    targets = []

    with torch.inference_mode():
        for batch in dataloader:
            xh, xa = batch["home"].to(device), batch["away"].to(device)
            mh, ma = batch["mask_home"].to(device), batch["mask_away"].to(device)

            ylogit = model(xh, xa, mh, ma)
            ypred = torch.argmax(ylogit, dim=1)

            preds.append(ypred.cpu())
            targets.append(batch["y"].squeeze(-1).long())
    
    preds = torch.cat(preds)
    targets = torch.cat(targets)

    return preds, targets