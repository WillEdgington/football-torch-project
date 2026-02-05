import torch

from typing import Callable, Dict, List, Any

from .trainer import Trial

EXPERIMENTDIR = "saved_models"

TrainFn = Callable[..., Dict[str, List[float]]]
ConstructorFn = Callable[
    [Trial], 
    Dict[
        str,
        torch.nn.Module|
        torch.optim.Optimizer|
        Dict[str,torch.utils.data.DataLoader]
    ]
]
EvaluatorFn = Callable[[Trial], Dict[str, Any]]