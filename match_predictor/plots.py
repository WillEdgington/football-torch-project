import torch
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

from .eval import evaluateClassificationModel, extractCalibrationSignals, binCalibration, computeECE

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

    cols = 1 + int(showAccuracy)
    fig, axes = plt.subplots(1, cols, figsize=(8 * cols, 8))

    if cols == 1:
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
                        title: str="",
                        device: torch.device="cuda" if torch.cuda.is_available() else "cpu"):
    preds, targets, _, _ = evaluateClassificationModel(model=model, 
                                                 dataloader=dataloader,
                                                 getProbs=False,
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
    ax.set_title(title + " Confusion Matrix" + (" (Normalized)" if normalise else ""))

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

def plotClassConfidenceHistogram(model: torch.nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 Ylabels: List[str]={0: "Home Win", 1: "Draw", 2: "Away Win"},
                                 bins: int=20,
                                 title: str="",
                                 device: torch.device="cuda" if torch.cuda.is_available() else "cpu"):
    _, targets, probs, _ = evaluateClassificationModel(model=model,
                                                    dataloader=dataloader,
                                                    getProbs=True,
                                                    device=device)
    
    classIndexes = list(Ylabels.keys())
    numClasses = len(classIndexes)

    if numClasses == 1:
        fig, axes = plt.subplots(1, 1, figsize=(6, 4))
        axes = [axes]
    else:
        cols = min(3, numClasses)
        rows = (2 + numClasses) // 3
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = axes.flatten()

    binEdges = torch.linspace(0.0, 1.0, steps=bins + 1)
    
    for ax, classIdx in zip(axes, classIndexes):
        classProbs = probs[:, classIdx]
        isClass = targets == classIdx

        correctProbs = classProbs[isClass]
        incorrectProbs = classProbs[~isClass]

        ax.hist(
            correctProbs.numpy(),
            bins=binEdges,
            alpha=0.9,
            label="Correct",
            density=True
        )

        ax.hist(
            incorrectProbs.numpy(),
            bins=binEdges,
            alpha=0.7,
            label="Incorrect",
            density=True
        )

        ax.set_title(Ylabels[classIdx])
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_xlim(0.0, 1.0)
        ax.legend()

    for ax in axes[numClasses:]:
        ax.axis("off")
    
    fig.suptitle(title + " Class Confidence Histogram", fontsize=14)
    plt.tight_layout()
    plt.show()

def plotReliabilityDiagram(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           bins: int=20,
                           title: str="",
                           device: torch.device="cuda" if torch.cuda.is_available() else "cpu"):
    preds, targets, probs, _ = evaluateClassificationModel(model=model,
                                                        dataloader=dataloader,
                                                        getProbs=True,
                                                        device=device)
    confidences, correct = extractCalibrationSignals(preds=preds, 
                                                     targets=targets, 
                                                     probs=probs)
    centers, acc, conf, count = binCalibration(confidences=confidences,
                                               correct=correct,
                                               bins=bins)
    ece = computeECE(binAcc=acc,
                     binConf=conf,
                     binCount=count)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(conf, acc, marker="o", label="Model")

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} Reliability Diagram (ECE = {ece:.3f})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()