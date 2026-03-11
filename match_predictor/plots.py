import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple

from experiments import TrialResult

def _resolveSplits(trialResult: TrialResult,
                   evalHash: str,
                   splits: List[str]|str|None) -> List[str]:
    available = trialResult.splits(evalHash)
    if splits is None:
        return available
    if isinstance(splits, str):
        splits = [splits]
    missing = [s for s in splits if s not in available]
    if missing:
        raise ValueError(f"Splits not found: {missing}. Available: {available}")
    return splits

def _axGrid(n: int,
            colWidth: int=6,
            rowHeight: int=5) -> Tuple[plt.Figure, List[plt.Axes]]:
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(colWidth * cols, rowHeight * rows))
    axes = np.array(axes).flatten().tolist()
    return fig, axes

def plotLoss(trialResult: TrialResult,
             show: bool=True,
             ax: plt.Axes|None=None):
    metrics = trialResult.metrics
    if metrics is None:
        raise ValueError("No training metrics found for this trial")
    
    owned = ax is None
    if owned:
        fig, ax = plt.subplots()
    
    ax.plot(metrics["train_loss"], label="Train")
    ax.plot(metrics["test_loss"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title("Loss") if owned else ax.set_ylabel("Loss")
    ax.legend()

    if owned and show:
        plt.tight_layout()
        plt.show()

def plotAccuracy(trialResult: TrialResult,
                 show: bool=True,
                 ax: plt.Axes|None=None):
    metrics = trialResult.metrics
    if metrics is None:
        raise ValueError("No training metrics found for this trial.")
    
    owned = ax is None
    if owned:
        fig, ax = plt.subplots()
    
    ax.plot(metrics["train_accuracy"], label="Train")
    ax.plot(metrics["test_accuracy"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title("Accuracy") if owned else ax.set_ylabel("Accuracy")
    ax.legend()

    if owned and show:
        plt.tight_layout()
        plt.show()

def plotTrainingCurves(trialResult: TrialResult,
                       show: bool=True,
                       trialID: int|None=None):
    metrics = trialResult.metrics
    if metrics is None:
        raise ValueError("No training metrics found for this trial.")
    
    hasAccuracy = "train_accuracy" in metrics
    cols = 1 + int(hasAccuracy)
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 5))
    if cols == 1:
        axes = [axes]
    
    plotLoss(trialResult=trialResult, show=False, ax=axes[0])
    if hasAccuracy:
        plotAccuracy(trialResult=trialResult, show=False, ax=axes[1])
    
    fig.suptitle(f"Training Curves - {trialResult._trial.path.name}" + f" (trial ID: {trialID})" if trialID else "")
    plt.tight_layout()
    if show: 
        plt.show()

def plotConfusionMatrix(trialResult: TrialResult,
                        evalHash: str,
                        splits: List[str]|str|None=None,
                        labels: List[str]=["Home Win", "Draw", "Away Win"],
                        normalise: bool=True,
                        show: bool=True,
                        trialID: int|None=None):
    resolved = _resolveSplits(trialResult=trialResult,
                              evalHash=evalHash,
                              splits=splits)
    fig, axes = _axGrid(len(resolved),
                        colWidth=6,
                        rowHeight=5)
    
    for ax, split in zip(axes, resolved):
        matrix = np.array(trialResult.getSplit(evalHash, split)["confusion_matrix"], dtype=float)

        if normalise:
            rowSums = matrix.sum(axis=1, keepdims=True)
            rowSums[rowSums == 0] = 1.0
            matrix = matrix / rowSums
        
        numClasses = len(labels)
        im = ax.imshow(matrix, 
                       cmap="Blues",
                       vmin=0,
                       vmax=1 if normalise else None)
        
        ax.set_xticks(range(numClasses))
        ax.set_yticks(range(numClasses))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{split.capitalize()}" + (" (Normalised)" if normalise else ""))

        for i in range(numClasses):
            for j in range(numClasses):
                val = matrix[i, j]
                text = f"{val:.2f}" if normalise else str(int(val))
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > 0.5 else "black")
        
        fig.colorbar(im, ax=ax)

    for ax in axes[len(resolved):]:
        ax.axis("off")
    
    fig.suptitle(f"Confusion Matrix - {trialResult._trial.path.name}" + f" (trial ID: {trialID})" if trialID else "")
    plt.tight_layout()
    if show:
        plt.show()

def plotReliabilityDiagram(trialResult: TrialResult,
                           evalHash: str,
                           splits: List[str]|str|None=None,
                           show: bool=True,
                           trialID: int|None=None):
    resolved = _resolveSplits(trialResult=trialResult,
                              evalHash=evalHash,
                              splits=splits)
    fig, axes = _axGrid(len(resolved),
                        colWidth=6,
                        rowHeight=6)
    
    for ax, split in zip(axes, resolved):
        data = trialResult.getSplit(evalHash, split)["calibration"]
        conf = np.array(data["bin_confidence"])
        acc = np.array(data["bin_accuracy"])
        count = np.array(data["bin_count"])
        ece = trialResult.getSplit(evalHash, split)["ece"]

        mask = count > 0
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(conf[mask], acc[mask], marker="o", label="Model")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{split.capitalize()} (ECE = {ece:.4f})")
        ax.legend()
        ax.grid(alpha=0.3)
    
    for ax in axes[len(resolved):]:
        ax.axis("off")
    
    fig.suptitle(f"Reliability Diagram - {trialResult._trial.path.name}" + f" (trial ID: {trialID})" if trialID else "")
    plt.tight_layout()
    if show:
        plt.show()

# def plotClassConfidenceHistogram(model: torch.nn.Module,
#                                  dataloader: torch.utils.data.DataLoader,
#                                  Ylabels: List[str]={0: "Home Win", 1: "Draw", 2: "Away Win"},
#                                  bins: int=20,
#                                  title: str="",
#                                  device: torch.device="cuda" if torch.cuda.is_available() else "cpu"):
#     _, targets, probs, _ = evaluateClassificationModel(model=model,
#                                                     dataloader=dataloader,
#                                                     getProbs=True,
#                                                     device=device)
    
#     classIndexes = list(Ylabels.keys())
#     numClasses = len(classIndexes)

#     if numClasses == 1:
#         fig, axes = plt.subplots(1, 1, figsize=(6, 4))
#         axes = [axes]
#     else:
#         cols = min(3, numClasses)
#         rows = (2 + numClasses) // 3
#         fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
#         axes = axes.flatten()

#     binEdges = torch.linspace(0.0, 1.0, steps=bins + 1)
    
#     for ax, classIdx in zip(axes, classIndexes):
#         classProbs = probs[:, classIdx]
#         isClass = targets == classIdx

#         correctProbs = classProbs[isClass]
#         incorrectProbs = classProbs[~isClass]

#         ax.hist(
#             correctProbs.numpy(),
#             bins=binEdges,
#             alpha=0.9,
#             label="Correct",
#             density=True
#         )

#         ax.hist(
#             incorrectProbs.numpy(),
#             bins=binEdges,
#             alpha=0.7,
#             label="Incorrect",
#             density=True
#         )

#         ax.set_title(Ylabels[classIdx])
#         ax.set_xlabel("Predicted Probability")
#         ax.set_ylabel("Density")
#         ax.set_xlim(0.0, 1.0)
#         ax.legend()

#     for ax in axes[numClasses:]:
#         ax.axis("off")
    
#     fig.suptitle(title + " Class Confidence Histogram", fontsize=14)
#     plt.tight_layout()
#     plt.show()