import torch

from typing import Dict, Any, List

from .trial import Trial
from .config import TrainFn, ConstructorFn
from utils.save import saveStates

class Trainer:
    def __init__(self,
                 trial: Trial,
                 train: TrainFn,
                 constructor: ConstructorFn):
        if trial.state is None or trial._definition is None:
            self.trial = Trial.load(path=trial.path)
        else:
            self.trial = trial

        self.train: TrainFn = train
        self.metrics: Dict[str, List[float]]|None = None
        self._loadTrainingParams()
        self._loadMetrics()

        loaded: Dict[
            str, 
            torch.nn.Module|
            torch.optim.Optimizer|
            Dict[str,torch.utils.data.DataLoader]
        ] = constructor(trial)
        self.dataloaders: Dict[str, torch.utils.data.DataLoader] = loaded["dataloaders"]
        self.model: torch.nn.Module = loaded["model"]
        self.optimizer: torch.optim.Optimizer = loaded["optimizer"]
        self.lossFn: torch.nn.Module = loaded["lossFn"]

    def run(self):
        if self.trial.isComplete():
            return self.model, self.metrics
        while self.epochsCompleted < self.maxEpoch:
            self.trial.state["status"] = "running"
            epochs = min(self.savepoint, self.maxEpoch - self.epochsCompleted)
            self.metrics = self.train(model=self.model,
                                      trainDataloader=self.dataloaders["train"],
                                      testDataloader=self.dataloaders.get("validation", self.dataloaders["test"]),
                                      lossFn=self.lossFn,
                                      optimizer=self.optimizer,
                                      epochs=epochs,
                                      seed=self.seed,
                                      results=self.metrics,
                                      calcAccuracy=self.trainParams.get("calcAccuracy", "true")=="true",
                                      enableAmp=self.trainParams.get("enableAmp", "true")=="true",
                                      gradClipping=self.trainParams.get("gradClipping", None),
                                      device=self.device)
            self.epochsCompleted += epochs
            self._saveCheckpoint()
        
        self.trial.state["status"] = "completed"
        self.trial.saveState()
        return self.model, self.metrics

    def _loadTrainingParams(self):
        definition, state = self.trial.getDefinition(), self.trial.getState()
        self.trainParams: Dict[str, Any] = definition["train"]
        self.seed: int = definition.get("seed", 42)
        self.device: str|torch.device = definition.get("device", "cpu")

        self.epochsCompleted: int = state["epochs_completed"]
        self.maxEpoch: int = state["max_epoch"]

        self.savepoint: int = self.trainParams.get("savepoint", 10)

    def _loadMetrics(self):
        if self.trial.metricsPath.exists():
            self.metrics = torch.load(self.trial.metricsPath)
        else:
            self.metrics = None

    def _saveMetrics(self):
        torch.save(self.metrics, self.trial.metricsPath)

    def _saveCheckpoint(self):
        if self.metrics is not None:
            self._saveMetrics()
        saveStates(stateName=f"MODEL_{self.epochsCompleted}_EPOCHS.pt",
                   stateDir=self.trial.modelPath,
                   model=self.model,
                   optimizer=self.optimizer)
        self.trial.state["epochs_completed"] = self.epochsCompleted
        self.trial.saveState()

# definition = {
#     "train": {
#         "epochs": epochs,
#         "savepoint": savepoint (epochs that will pass into the train function instead of "epochs"),
#         other training parameters (this is parameters we will pass into the train method)
#     },
#     "model": {
#         "class": torch.nn.Module class name,
#         other model parameters (these are parameters that will be passed into the model class)
#     },
#     "data": {
#         data parameters (these will be things that are passed into the data construction)
#     },
#     "seed": seed,
#     "device": device
# }