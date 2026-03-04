from experiments import GridTrialScheduler, Experiment, ExperimentResults, Evaluator

from .train import train
from .constructor import constructMatchPredictorTrial
from .config import BASEDEFINITION, SWEEP, EVALDEFINITIONBIG5, SAVEDMODELSDIR
from .eval import evaluateMatchPredictorModel

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

    results = ExperimentResults(root=SAVEDMODELSDIR)
    resultsDf = results.toDataFrame(evalHash=evaluator.evalHash)
    print(resultsDf[resultsDf["test.ece"] <= 0.05].sort_values(by="test.ece").head(20))