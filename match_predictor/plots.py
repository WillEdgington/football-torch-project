import torch
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

from .eval import evaluateClassificationModel

def plotLoss(results: Dict[str, list], show: bool=True, ax: plt.Axes|None=None) -> Tuple[plt.Figure, plt.Axes]|None:
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(results["train_loss"], label="Train")
    ax.plot(results["test_loss"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title("Loss")
    ax.legend()

    if show:
        plt.show()

def plotAccuracy(results: Dict[str, list], show: bool=True, ax: plt.Axes|None=None):
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(results["train_accuracy"], label="Train")
    ax.plot(results["test_accuracy"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title("Accuracy")
    ax.legend()

    if show:
        plt.show()

def plotResults(results: Dict[str, float], title: str):
    showAccuracy = "train_accuracy" in results

    rows = 1 + int(showAccuracy)
    fig, axes = plt.subplots(1, rows, figsize=(8 * rows, 8))

    if rows == 1:
        axes = [axes]

    plotLoss(results=results, show=False, ax=axes[0])
    if showAccuracy:
        plotAccuracy(results=results, show=False, ax=axes[1])
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plotConfusionMatrix(model: torch.nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        Ylabels: List[str]=["Home Win", "Draw", "Away Win"],
                        normalise: bool=True,
                        device: torch.device="cuda" if torch.cuda.is_available() else "cpu"):
    preds, targets = evaluateClassificationModel(model=model, 
                                                 dataloader=dataloader,
                                                 device=device)
    
    numClasses = len(Ylabels)
    conf = torch.zeros((numClasses, numClasses), dtype=torch.int32)

    for t, p in zip(targets, preds):
        conf[t, p] += 1
    
    conf = conf.numpy().astype(float)

    if normalise:
        rowSums = conf.sum(axis=1, keepdims=True)
        rowSums[rowSums == 0] = 1.0
        conf = conf / rowSums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf, cmap="Blues")

    ax.set_xticks(range(numClasses))
    ax.set_yticks(range(numClasses))
    ax.set_xticklabels(Ylabels)
    ax.set_yticklabels(Ylabels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix" + (" (Normalized row-wise)" if normalise else ""))

    for i in range(numClasses):
        for j in range(numClasses):
            val = conf[i, j]
            text = f"{val:.2f}" if normalise else str(conf[i, j].item())
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()