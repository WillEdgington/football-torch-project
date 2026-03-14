import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List, Tuple, Literal

from experiments import TrialResult, ExperimentResults

from .config import CLASSLABELS

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

def _trialTitle(trialResult: TrialResult,
                trialID: int|None=None) -> str:
    return trialResult._trial.path.name + (f" (trial ID: {trialID})" if trialID else "")

def plotLoss(trialResult: TrialResult,
             show: bool=True,
             ax: plt.Axes|None=None,
             trialID: str|None=None) -> plt.Figure:
    metrics = trialResult.metrics
    if metrics is None:
        raise ValueError("No training metrics found for this trial")
    
    owned = ax is None
    if owned:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    ax.plot(metrics["train_loss"], label="Train")
    ax.plot(metrics["test_loss"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Loss - {_trialTitle(trialResult, trialID)}") if owned else ax.set_ylabel("Loss")
    ax.legend()

    if owned:
        plt.tight_layout()
        if show:
            plt.show()
    return fig

def plotAccuracy(trialResult: TrialResult,
                 show: bool=True,
                 ax: plt.Axes|None=None,
                 trialID: str|None=None) -> plt.Figure:
    metrics = trialResult.metrics
    if metrics is None:
        raise ValueError("No training metrics found for this trial.")
    
    owned = ax is None
    if owned:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    
    ax.plot(metrics["train_accuracy"], label="Train")
    ax.plot(metrics["test_accuracy"], label="Test")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Accuracy - {_trialTitle(trialResult, trialID)}") if owned else ax.set_ylabel("Accuracy")
    ax.legend()

    if owned:
        plt.tight_layout()
        if show:
            plt.show()
    return fig

def plotTrainingCurves(trialResult: TrialResult,
                       show: bool=True,
                       trialID: int|None=None,
                       suptitle: str|None="default") -> plt.Figure:
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
    
    if suptitle is not None:
        fig.suptitle(f"Training Curves - {_trialTitle(trialResult, trialID)}" if suptitle == "default" else suptitle)

    plt.tight_layout()
    if show: 
        plt.show()
    return fig

def plotConfusionMatrix(trialResult: TrialResult,
                        evalHash: str,
                        splits: List[str]|str|None=None,
                        labels: List[str]=CLASSLABELS,
                        normalise: bool=True,
                        show: bool=True,
                        trialID: int|None=None,
                        suptitle: str|None="default") -> plt.Figure:
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
    
    if suptitle is not None:
        fig.suptitle(f"Confusion Matrix - {_trialTitle(trialResult, trialID)}" if suptitle == "default" else suptitle)

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plotReliabilityDiagram(trialResult: TrialResult,
                           evalHash: str,
                           splits: List[str]|str|None=None,
                           show: bool=True,
                           trialID: int|None=None,
                           suptitle: str|None="default") -> plt.Figure:
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
    
    if suptitle is not None:
        fig.suptitle(f"Reliability Diagram - {_trialTitle(trialResult, trialID)}" if suptitle == "default" else suptitle)

    plt.tight_layout()
    if show:
        plt.show()
    return fig

def plotTrialSummary(trialResult: TrialResult,
                     evalHash: str|None=None,
                     splits: List[str]|str|None=None,
                     labels: List[str]=CLASSLABELS,
                     normalise: bool=True,
                     trialID: int|None=None,
                     show: bool=True,
                     suptitle: str|None="default",
                     trainmetrics: bool=True,
                     confmat: bool=True,
                     reliability: bool=True) -> List[plt.Figure]:
    totalPlots = int(trainmetrics) + int(confmat) + int(reliability)
    if totalPlots == 0:
        raise ValueError("Atleast one of trainmetrics, confmat, or reliability must be True")
    if evalHash is None and (confmat or reliability):
        raise ValueError("For confusion matrix or reliability plots you must have an evaluator hash (evalHash).")
    
    figs = []

    if trainmetrics:
        figs.append(plotTrainingCurves(trialResult=trialResult,
                                       trialID=trialID,
                                       suptitle=suptitle,
                                       show=False))

    if confmat:
        figs.append(plotConfusionMatrix(trialResult=trialResult,
                                        evalHash=evalHash,
                                        splits=splits,
                                        labels=labels,
                                        normalise=normalise,
                                        trialID=trialID,
                                        suptitle=suptitle,
                                        show=False))
    
    if reliability:
        figs.append(plotReliabilityDiagram(trialResult=trialResult,
                                           evalHash=evalHash,
                                           splits=splits,
                                           trialID=trialID,
                                           suptitle=suptitle,
                                           show=False))
        
    if show:
        plt.show()
    return figs

def plotExperimentMetricScatter(xlabel: str,
                                ylabel: str,
                                experimentResults: ExperimentResults|None=None,
                                evalHash: str|None=None,
                                df: pd.DataFrame|None=None,
                                mode: Literal["scatter", "errorbar", "scatter+fit", "errorbar+fit"]="scatter",
                                ax: plt.Axes|None=None,
                                show: bool=True) -> plt.Figure:
    if df is None and (experimentResults is None or evalHash is None):
        raise ValueError("You must input a DataFrame (df) or an ExperimentResults object (experimentResults) and evaluator hash (evalHash)")
    
    if df is None:
        df = experimentResults.toDataFrame(evalHash=evalHash)
    if xlabel not in df.columns:
        raise ValueError(f"Column '{xlabel}' not in DataFrame")
    if ylabel not in df.columns:
        raise ValueError(f"Column '{ylabel}' not in DataFrame")
    
    owned = ax is None
    if owned:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    x = df[xlabel]
    y = df[ylabel]

    if mode.startswith("scatter"):
        ax.scatter(df[xlabel], df[ylabel], alpha=0.7)
        for trial_id, row in df.iterrows():
            ax.annotate(str(trial_id), 
                        (row[xlabel], row[ylabel]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7)
    elif mode.startswith("errorbar"):
        grouped = df.groupby(xlabel)[ylabel]
        means = grouped.mean()
        stds = grouped.std().fillna(0)
        ax.errorbar(means.index, 
                    means.values, 
                    yerr=stds.values,
                    fmt="o",
                    capsize=4,
                    color="steelblue")
    if mode.endswith("+fit"):
        coeffs = np.polyfit(x, y, 1)
        xLine = np.linspace(x.min(), x.max(), 200)
        ax.plot(xLine,
                np.polyval(coeffs, xLine),
                color="tomato",
                linewidth=1.5,
                label=f"fit: f(x)={coeffs[0]:.3f}x + {coeffs[1]:.3f}")
        ax.legend()
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs {xlabel}")
    ax.grid(alpha=0.3)

    if owned:
        plt.tight_layout()
        if show:
            plt.show()
    return fig

def plotExperimentMetricBar(metric: str,
                            experimentResults: ExperimentResults|None=None,
                            evalHash: str|None=None,
                            df: pd.DataFrame|None=None,
                            ascending: bool=False,
                            topN: int|None=None,
                            ax: plt.Axes|None=None,
                            show: bool=True) -> plt.Figure:
    if df is None and (experimentResults is None or evalHash is None):
        raise ValueError("You must input a DataFrame (df) or an ExperimentResults object (experimentResults) and evaluator hash (evalHash)")
    
    if df is None:
        df = experimentResults.toDataFrame(evalHash=evalHash)
    if metric not in df.columns:
        raise ValueError(f"Column '{metric}' not in DataFrame.")

    ranked = df[metric].dropna().sort_values(ascending=ascending)
    if topN is not None:
        ranked = ranked.head(topN)

    owned = ax is None
    if owned:
        fig, ax = plt.subplots(figsize=(max(6, len(ranked) * 0.6), 5))
    else:
        fig = ax.get_figure()

    bars = ax.bar(ranked.index.astype(str), ranked.values, color="steelblue")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel(metric)
    ax.set_title(f"Trials ranked by {metric}" + (f" (top {topN})" if topN else ""))
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, ranked.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

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