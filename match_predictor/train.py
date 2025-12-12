import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def trainStep(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              calcAccuracy: bool=False,
              enableAmp: bool=True,
              gradClipping: None|float=None,
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float|None]:
    model.train()

    classification = isinstance(lossFn, torch.nn.CrossEntropyLoss)
    trainLoss = 0.0
    scaler = torch.amp.GradScaler(device=device, enabled=(device=="cuda" and enableAmp))

    for batch in dataloader:
        xh, xa = batch["home"].to(device), batch["away"].to(device)
        mh, ma = batch["mask_home"].to(device), batch["mask_away"].to(device)
        Y = batch["y"].to(device)
        if classification:
            Y = Y.squeeze(-1).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device, enabled=(device=="cuda" and enableAmp)):
            ylogit = model(xh, xa, mh, ma)
            loss = lossFn(ylogit, Y)

        scaler.scale(loss).backward()
        if gradClipping is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClipping)
        scaler.step(optimizer)
        scaler.update()

        trainLoss += loss.item()

    return trainLoss / len(dataloader), None

def testStep(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             calcAccuracy: bool=True,
             enableAmp: bool=True,
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float|None]:
    model.eval()

    classification = isinstance(lossFn, torch.nn.CrossEntropyLoss)
    testLoss = 0.0

    with torch.inference_mode():
        for batch in dataloader:
            xh, xa = batch["home"].to(device), batch["away"].to(device)
            mh, ma = batch["mask_home"].to(device), batch["mask_away"].to(device)
            Y = batch["y"].to(device)
            if classification:
                Y = Y.squeeze(-1).long()

            with torch.amp.autocast(device_type=device, enabled=(device=="cuda" and enableAmp)):
                ylogit = model(xh, xa, mh, ma)
                loss = lossFn(ylogit, Y)
            testLoss += loss.item()
    
    return testLoss / len(dataloader), None

def train(model: torch.nn.Module,
          trainDataloader: torch.utils.data.DataLoader,
          testDataloader: torch.utils.data.DataLoader,
          lossFn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int=5,
          results: Dict[str, List[float]]|None=None,
          calcAccuracy: bool=False,
          enableAmp: bool=True,
          gradClipping: None|float=None,
          device: str="cuda" if torch.cuda.is_available() else "cpu") -> Dict[str, List[float]]:
    model.to(device)

    initialEpoch = 1
    if results is not None: 
        initialEpoch += len(results["train_loss"])
    else:
        results = {
            "train_loss": [],
            "test_loss": [],
        }
        if calcAccuracy:
            results["train_accuracy"] = []
            results["test_accuracy"] = []

    for epoch in tqdm(range(epochs)):
        trainLoss, trainAccuracy = trainStep(model=model,
                                             dataloader=trainDataloader,
                                             lossFn=lossFn,
                                             optimizer=optimizer,
                                             calcAccuracy=calcAccuracy,
                                             enableAmp=enableAmp,
                                             gradClipping=gradClipping,
                                             device=device)
        results["train_loss"].append(trainLoss)
        if calcAccuracy: results["train_accuracy"].append(trainAccuracy)

        testLoss, testAccuracy = testStep(model=model,
                                          dataloader=testDataloader,
                                          lossFn=lossFn,
                                          calcAccuracy=calcAccuracy,
                                          enableAmp=enableAmp,
                                          device=device)
        results["test_loss"].append(testLoss)
        if calcAccuracy: results["test_accuracy"].append(testAccuracy)

        print(f"\nEpochs: {epoch+initialEpoch} | (Loss) Train: {trainLoss:.4f}, Test: {testLoss:.4f}" 
              + (f" | (Accuracy) Train: {trainAccuracy:.2f}, Test: {testAccuracy:.2f}" if calcAccuracy else ""))
    
    return results