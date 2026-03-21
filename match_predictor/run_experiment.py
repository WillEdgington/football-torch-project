from experiments import Evaluator, Experiment, GridTrialScheduler

from .config import BASEDEFINITION, EVALDEFINITIONBIG5, SAVEDMODELSDIR, SWEEP
from .constructor import constructMatchPredictorTrial
from .eval import evaluateMatchPredictorModel
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
    experiment.run()
