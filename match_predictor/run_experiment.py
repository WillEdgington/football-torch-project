from experiments import GridTrialScheduler, Experiment

from .train import train
from .constructor import constructMatchPredictorTrial
from .config import BASEDEFINITION, SWEEP, SAVEDMODELSDIR

if __name__=="__main__":
    trialScheduler = GridTrialScheduler(baseDefinition=BASEDEFINITION,
                                        sweep=SWEEP)
    experiment = Experiment(root=SAVEDMODELSDIR,
                            scheduler=trialScheduler,
                            train=train,
                            constructer=constructMatchPredictorTrial,
                            evaluator=None) # evaluator needs implementing
    experiment.run()