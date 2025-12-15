import matplotlib.pyplot as plt

from typing import Dict, Tuple

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