from experiments import (
    Evaluator,
    Experiment,
    ExperimentResults,
    GridTrialScheduler,
    addCompositeScore,
)

from .config import (
    BASEDEFINITION,
    COMPASCENDING,
    COMPWEIGHTS,
    EVALDEFINITIONBIG5,
    SAVEDMODELSDIR,
    SWEEP,
)
from .constructor import constructMatchPredictorTrial
from .eval import evaluateMatchPredictorModel
from .plots import (
    plotExperimentMetricBar,
    plotExperimentMetricScatter,
    plotTrialSummary,
)
from .train import train

if __name__ == "__main__":
    evaluator = Evaluator(
        eval=evaluateMatchPredictorModel,
        constructor=constructMatchPredictorTrial,
        evalDefinition=EVALDEFINITIONBIG5,
    )
    trialScheduler = GridTrialScheduler(baseDefinition=BASEDEFINITION, sweep=SWEEP)
    experiment = Experiment(
        root=SAVEDMODELSDIR,
        scheduler=trialScheduler,
        train=train,
        constructer=constructMatchPredictorTrial,
        evaluator=evaluator,
    )
    experiment.eval()
    evalHash = evaluator.evalHash

    results = ExperimentResults(root=SAVEDMODELSDIR)
    resultsDf = results.toDataFrame(evalHash=evalHash)
    filteredDf = resultsDf[resultsDf["test.ece"] <= 0.05]
    filteredDf = addCompositeScore(
        df=filteredDf,
        weights=COMPWEIGHTS,
        ascending=COMPASCENDING,
        colName="composite_score",
    ).sort_values(by="composite_score", ascending=False)
    print(
        filteredDf.head(20)[
            [
                "model.featExtractorExpansionFFN",
                "model.featExtractorActivationFFN",
                "model.activationMLP",
                "data.batchSize",
                "data.seqLen",
                "lossFn.label_smoothing",
                "test.ece",
                "test.accuracy",
                "test.loss",
                "composite_score",
            ]
        ]
    )
    trialID = filteredDf.index[0]
    plotExperimentMetricScatter(
        experimentResults=None,
        evalHash=None,
        df=filteredDf,
        xlabel="data.batchSize",
        ylabel="composite_score",
        mode="errorbar+fit",
    )
    plotExperimentMetricBar(
        metric="composite_score",
        experimentResults=None,
        evalHash=None,
        df=filteredDf,
        ascending=False,
        topN=10,
        show=True,
    )
    trialResult = results.getTrial(trial_id=trialID)
    fig = plotTrialSummary(
        trialResult=trialResult,
        evalHash=evalHash,
        normalise=True,
        trialID=trialID,
        show=True,
    )
