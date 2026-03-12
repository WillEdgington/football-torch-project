from experiments import GridTrialScheduler, Experiment, ExperimentResults, Evaluator

from .train import train
from .constructor import constructMatchPredictorTrial
from .config import BASEDEFINITION, SWEEP, EVALDEFINITIONBIG5, SAVEDMODELSDIR
from .eval import evaluateMatchPredictorModel
from .plots import plotTrainingCurves, plotConfusionMatrix, plotReliabilityDiagram, plotExperimentMetricScatter, plotExperimentMetricBar

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
    filteredDf = resultsDf[resultsDf["test.ece"] <= 0.05].sort_values(by="test.accuracy", ascending=False).head(20)
    print(filteredDf)
    plotExperimentMetricScatter(experimentResults=None,
                                evalHash=None,
                                df=filteredDf,
                                xlabel="data.batchSize",
                                ylabel="test.accuracy",
                                mode="errorbar+fit")
    plotExperimentMetricBar(metric="test.accuracy",
                            experimentResults=None,
                            evalHash=None,
                            df=filteredDf,
                            ascending=False,
                            topN=10,
                            show=True)
    trialID = 352
    trialResult = results.getTrial(trial_id=trialID)
    # print(trialResult.definition)
    # print(trialResult.getEval(evalHash))
    plotTrainingCurves(trialResult=trialResult,
                       show=True,
                       trialID=trialID)
    plotConfusionMatrix(trialResult=trialResult,
                        evalHash=evalHash,
                        splits=None,
                        normalise=True,
                        show=True,
                        trialID=trialID)
    plotReliabilityDiagram(trialResult=trialResult,
                           evalHash=evalHash,
                           splits=None,
                           show=True,
                           trialID=trialID)