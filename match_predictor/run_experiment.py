from experiments import GridTrialScheduler, Experiment, Evaluator

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
                            evaluator=evaluator) # evaluator needs implementing
    experiment.eval()
    experiment.run()