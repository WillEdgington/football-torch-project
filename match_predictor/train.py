import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from tensor_pipeline import Normaliser

def correctPredictions(ylogit: torch.Tensor, 
                      Y: torch.Tensor, 
                      hgidx: int|None, 
                      agidx: int|None=None,
                      goalsStd: float|None=None,
                      goalsMean: float|None=None,
                      normaliserName: str="numeric_normaliser.json") -> float:
    if hgidx is None or agidx is None:
        pred = torch.argmax(ylogit, dim=1)
        actual = Y
    else:
        if goalsStd is None or goalsMean is None:
            with Normaliser(train=False, fileName=normaliserName) as nrm:
                goalsParams = nrm.params["goals"]
                goalsStd, goalsMean = goalsParams["std"], goalsParams["mean"]

        homeGoalsPred = (goalsStd * ylogit[:, hgidx] + goalsMean).round()
        awayGoalsPred = (goalsStd * ylogit[:, agidx] + goalsMean).round()
        pred = homeGoalsPred - awayGoalsPred
        actual = Y[:, hgidx] - Y[:, agidx]
        
        pred = torch.sign(pred)
        actual = torch.sign(actual)

    return (pred == actual).sum().item()

def trainStep(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              calcAccuracy: bool=False,
              enableAmp: bool=True,
              gradClipping: None|float=None,
              goalsStd: float|None=None,
              goalsMean: float|None=None,
              normaliserName: str="numeric_normaliser.json",
              device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float|None]:
    model.train()

    classification = isinstance(lossFn, torch.nn.CrossEntropyLoss)

    if calcAccuracy:
        trainCorrect = 0.0
        trainSamples = 0.0
        hgidx, agidx = None, None
        if not classification:
            yCols = dataloader.dataset.yCols
            for i, col in enumerate(yCols):
                if col.startswith("home_goals"):
                    hgidx = i
                if col.startswith("away_goals"):
                    agidx = i
            calcAccuracy = not (hgidx is None or agidx is None)

            if goalsStd is None or goalsMean is None:
                with Normaliser(train=False, fileName=normaliserName) as nrm:
                    goalsParams = nrm.params["goals"]
                    goalsStd, goalsMean = goalsParams["std"], goalsParams["mean"]

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

        if calcAccuracy:
            correct = correctPredictions(ylogit=ylogit, Y=Y, hgidx=hgidx, agidx=agidx, goalsStd=goalsStd, goalsMean=goalsMean)
            trainCorrect += correct
            trainSamples += Y.size(0)

        scaler.scale(loss).backward()
        if gradClipping is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClipping)
        scaler.step(optimizer)
        scaler.update()

        trainLoss += loss.item()

    return trainLoss / len(dataloader), trainCorrect * 100.0 / trainSamples if calcAccuracy else None

def testStep(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             lossFn: torch.nn.Module,
             calcAccuracy: bool=True,
             enableAmp: bool=True,
             goalsStd: float|None=None,
             goalsMean: float|None=None,
             normaliserName: str="numeric_normaliser.json",
             device: torch.device="cuda" if torch.cuda.is_available() else "cpu") -> Tuple[float, float|None]:
    model.eval()

    classification = isinstance(lossFn, torch.nn.CrossEntropyLoss)
    
    if calcAccuracy:
        testCorrect = 0.0
        testSamples = 0.0
        hgidx, agidx = None, None
        if not classification:
            yCols = dataloader.dataset.yCols
            for i, col in enumerate(yCols):
                if col.startswith("home_goals"):
                    hgidx = i
                if col.startswith("away_goals"):
                    agidx = i
            calcAccuracy = not (hgidx is None or agidx is None)

            if goalsStd is None or goalsMean is None:
                with Normaliser(train=False, fileName=normaliserName) as nrm:
                    goalsParams = nrm.params["goals"]
                    goalsStd, goalsMean = goalsParams["std"], goalsParams["mean"]

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

            if calcAccuracy:
                correct = correctPredictions(ylogit=ylogit, Y=Y, hgidx=hgidx, agidx=agidx, goalsStd=goalsStd, goalsMean=goalsMean)
                testCorrect += correct
                testSamples += Y.size(0)
    
    return testLoss / len(dataloader), testCorrect * 100.0 / testSamples if calcAccuracy else None

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
          normaliserName: str="numeric_normaliser.json",
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
    

    classification = isinstance(lossFn, torch.nn.CrossEntropyLoss)
    goalsStd, goalsMean = None, None
    if calcAccuracy and not classification:
        with Normaliser(train=False, fileName=normaliserName) as nrm:
            goalsParams = nrm.params["goals"]
            goalsStd, goalsMean = goalsParams["std"], goalsParams["mean"]
            

    for epoch in tqdm(range(epochs)):
        trainLoss, trainAccuracy = trainStep(model=model,
                                             dataloader=trainDataloader,
                                             lossFn=lossFn,
                                             optimizer=optimizer,
                                             calcAccuracy=calcAccuracy,
                                             enableAmp=enableAmp,
                                             gradClipping=gradClipping,
                                             goalsStd=goalsStd,
                                             goalsMean=goalsMean,
                                             device=device)
        results["train_loss"].append(trainLoss)
        if calcAccuracy: results["train_accuracy"].append(trainAccuracy)

        testLoss, testAccuracy = testStep(model=model,
                                          dataloader=testDataloader,
                                          lossFn=lossFn,
                                          calcAccuracy=calcAccuracy,
                                          enableAmp=enableAmp,
                                          goalsStd=goalsStd,
                                          goalsMean=goalsMean,
                                          device=device)
        results["test_loss"].append(testLoss)
        if calcAccuracy: results["test_accuracy"].append(testAccuracy)

        print(f"\nEpochs: {epoch+initialEpoch} | (Loss) Train: {trainLoss:.4f}, Test: {testLoss:.4f}" 
              + (f" | (Accuracy) Train: {trainAccuracy:.2f}, Test: {testAccuracy:.2f}" if calcAccuracy else ""))
    
    return results