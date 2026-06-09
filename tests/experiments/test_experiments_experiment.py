from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.experiment import Experiment


@pytest.fixture
def mock_experiment_setup(tmp_path):
    with (
        patch.object(Experiment, "_loadTrials", return_value=[]),
        patch.object(Experiment, "_loadDefinitionHashes", return_value=set()),
    ):
        yield tmp_path


def test_Experiment_initPathAndEvaluatorNormalisation(mock_experiment_setup):
    rootStr = str(mock_experiment_setup / "exp_root")
    mockEvaluator = MagicMock()

    expOneEval = Experiment(
        root=rootStr,
        scheduler=MagicMock,
        train=MagicMock,
        constructer=MagicMock(),
        evaluator=mockEvaluator,
    )
    assert isinstance(expOneEval.root, Path)
    assert expOneEval.evaluator == [mockEvaluator]
    assert (expOneEval.root / "trials").exists()

    expNoEval = Experiment(
        root=rootStr,
        scheduler=MagicMock,
        train=MagicMock,
        constructer=MagicMock(),
        evaluator=None,
    )
    assert expNoEval.evaluator == []


@patch.object(Experiment, "_saveTrials")
def test_Experiment_evalTrialManagesUniqueAndDuplicateHashes(
    mock_save, mock_experiment_setup
):
    evalA = MagicMock()
    evalA.run.return_value = "hash_alpha"
    evalB = MagicMock()
    evalB.run.return_value = "hash_beta"

    exp = Experiment(
        root=mock_experiment_setup,
        scheduler=MagicMock,
        train=MagicMock,
        constructer=MagicMock(),
        evaluator=[evalA, evalB],
    )
    exp.trials = [{"evals": ["hash_alpha"]}]

    hashes = exp.evalTrial(trial=MagicMock(), i=0)

    assert hashes == ["hash_alpha", "hash_beta"]
    assert exp.trials[0]["evals"] == ["hash_alpha", "hash_beta"]
    mock_save.assert_called_once()


@patch("experiments.experiment.Trial.load")
@patch.object(Experiment, "_saveTrials")
def test_Experiment_bultEvalSkipsIncompleteOrPreviouslyProcessedRecords(
    mock_save, mock_trial_load, mock_experiment_setup
):
    evaluator = MagicMock()
    evaluator.evalHash = "hash_delta"
    evaluator.run.return_value = "hash_delta"

    exp = Experiment(
        root=mock_experiment_setup,
        scheduler=MagicMock(),
        train=MagicMock(),
        constructer=MagicMock(),
        evaluator=[evaluator],
    )
    exp.trials = [
        {"path": "/trial/0", "evals": []},
        {"path": "/trial/1", "evals": []},
        {"path": "/trial/2", "evals": ["hash_delta"]},
    ]

    mockTrial0 = MagicMock()
    mockTrial0.isComplete.return_value = True
    mockTrial1 = MagicMock()
    mockTrial1.isComplete.return_value = False
    mockTrial2 = MagicMock()
    mockTrial2.isComplete.return_value = True

    mock_trial_load.side_effect = [mockTrial0, mockTrial1, mockTrial2]
    exp.eval()

    evaluator.run.assert_called_once_with(mockTrial0)
    assert exp.trials[0]["evals"] == ["hash_delta"]
    assert exp.trials[1]["evals"] == []
    mock_save.assert_called_once()


@patch("experiments.experiment.Trainer")
def test_Experiment_runTerminatesSafelyAndCoordinatesDownstreamTasks(
    mock_trainer_cls, mock_experiment_setup
):
    evaluator = MagicMock()
    exp = Experiment(
        root=mock_experiment_setup,
        scheduler=MagicMock(),
        train=MagicMock(),
        constructer=MagicMock(),
        evaluator=[evaluator],
    )
    mockTrialTarget = MagicMock()

    exp._prepareTrial = MagicMock(side_effect=[(mockTrialTarget, 0), (None, None)])
    exp.evalTrial = MagicMock()

    mockTrainerInstance = MagicMock()
    mock_trainer_cls.return_value = mockTrainerInstance
    exp.run()

    mock_trainer_cls.assert_called_once_with(
        trial=mockTrialTarget, train=exp.train, constructor=exp.constructor
    )
    mockTrainerInstance.run.assert_called_once()

    exp.evalTrial.assert_called_once_with(mockTrialTarget, 0)
