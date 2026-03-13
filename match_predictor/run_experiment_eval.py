from experiments import GridTrialScheduler, Experiment, ExperimentResults, Evaluator, addCompositeScore

from .train import train
from .constructor import constructMatchPredictorTrial
from .config import BASEDEFINITION, SWEEP, EVALDEFINITIONBIG5, SAVEDMODELSDIR, COMPWEIGHTS, COMPASCENDING
from .eval import evaluateMatchPredictorModel
from .plots import plotTrainingCurves, plotConfusionMatrix, plotReliabilityDiagram, plotExperimentMetricScatter, plotExperimentMetricBar, plotTrialSummary

if __name__=="__main__":
    evaluator = Evaluator(eval=evaluateMatchPredictorModel,
                          constructor=constructMatchPredictorTrial,
                          evalDefinition=EVALDEFINITIONBIG5)
    trialScheduler = GridTrialScheduler(baseDefinition=BASEDEFINITION,
                                        sweep=SWEEP)
    experiment = Experiment(root=SAVEDMODELSDIR,
                            scheduler=trialScheduler,
                            train=train,
                            constructer=constructMatchPredictorTrial,
                            evaluator=evaluator)
    experiment.eval()
    evalHash = evaluator.evalHash

    results = ExperimentResults(root=SAVEDMODELSDIR)
    resultsDf = results.toDataFrame(evalHash=evalHash)
    filteredDf = resultsDf[resultsDf["test.ece"] <= 0.05]
    filteredDf = addCompositeScore(df=filteredDf,
                                   weights=COMPWEIGHTS,
                                   ascending=COMPASCENDING,
                                   colName="composite_score")
    print(filteredDf.sort_values(by="composite_score", ascending=False).head(20)\
          [["model.featExtractorActivationFFN", "model.activationMLP",
            "data.batchSize", "data.seqLen",
            "lossFn.label_smoothing", 
            "test.ece", "test.accuracy", "test.loss", 
            "composite_score"]])
    plotExperimentMetricScatter(experimentResults=None,
                                evalHash=None,
                                df=filteredDf,
                                xlabel="data.seqLen",
                                ylabel="composite_score",
                                mode="errorbar+fit")
    plotExperimentMetricBar(metric="composite_score",
                            experimentResults=None,
                            evalHash=None,
                            df=filteredDf,
                            ascending=False,
                            topN=10,
                            show=True)
    trialID = 369
    trialResult = results.getTrial(trial_id=trialID)
    # print(trialResult.definition)
    # print(trialResult.getEval(evalHash))
    # plotTrainingCurves(trialResult=trialResult,
    #                    show=True,
    #                    trialID=trialID)
    # plotConfusionMatrix(trialResult=trialResult,
    #                     evalHash=evalHash,
    #                     splits=None,
    #                     normalise=True,
    #                     show=True,
    #                     trialID=trialID)
    # plotReliabilityDiagram(trialResult=trialResult,
    #                        evalHash=evalHash,
    #                        splits=None,
    #                        show=True,
    #                        trialID=trialID)
    fig = plotTrialSummary(trialResult=trialResult,
                           evalHash=evalHash,
                           normalise=True,
                           trialID=trialID,
                           show=True)