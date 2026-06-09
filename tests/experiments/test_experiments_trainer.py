from unittest.mock import MagicMock, patch

import pytest

from experiments.trainer import Trainer


@pytest.fixture
def mock_trial():
    """Provides a minimal, framework-agnostic Trial double."""
    trial = MagicMock()
    trial.state = {"status": "created", "epochs_completed": 0, "max_epoch": 20}
    trial._definition = {"train": {"savepoint": 10}}
    trial.getDefinition.return_value = trial._definition
    trial.getState.return_value = trial.state
    trial.metricsPath.exists.return_value = False
    trial.isComplete.return_value = False
    return trial


@pytest.fixture
def mock_constructor():
    constructor = MagicMock()
    constructor.return_value = {
        "dataloaders": {"train": "mock_loader", "test": "mock_loader"},
        "model": "opaque_model_object",
        "optimizer": "opaque_optimizer_object",
        "lossFn": "opaque_loss_object",
    }
    return constructor


def test_Trainer_initWithFullyLoadedTrial(mock_trial, mock_constructor):
    trainFn = MagicMock()
    trainer = Trainer(mock_trial, trainFn, mock_constructor)

    assert trainer.trial == mock_trial
    mock_constructor.assert_called_once_with(mock_trial)


@patch("experiments.trainer.Trial.load")
def test_Trainer_initLifecycle(mock_trial_load, mock_trial, mock_constructor):
    trainer = Trainer(mock_trial, MagicMock(), mock_constructor)
    assert trainer.trial == mock_trial
    mock_trial_load.assert_not_called()

    unloadedTrial = MagicMock()
    unloadedTrial.state = None
    unloadedTrial._definition = None
    unloadedTrial.path = "/mock/path"
    mock_trial_load.return_value = mock_trial

    trainerFallback = Trainer(unloadedTrial, MagicMock(), mock_constructor)
    assert trainerFallback.trial == mock_trial
    mock_trial_load.assert_called_once_with(path="/mock/path")


def test_Trainer_runBypassesLoopIfAlreadyComplete(mock_trial, mock_constructor):
    mock_trial.isComplete.return_value = True
    trainFn = MagicMock()

    trainer = Trainer(mock_trial, trainFn, mock_constructor)
    model, _ = trainer.run()

    assert model == "opaque_model_object"
    trainFn.assert_not_called()


@patch("experiments.trainer.torch.save")
@patch("experiments.trainer.saveStates")
def test_Trainer_runOrchestratesLoopIncrementsAndSavesState(
    mock_save_states, mock_torch_save, mock_trial, mock_constructor
):
    trainFn = MagicMock(return_value={"loss": [0.5]})

    # max_epoch = 20, savepoint = 10. trainFn should call twice
    trainer = Trainer(mock_trial, trainFn, mock_constructor)
    _, _ = trainer.run()

    assert trainFn.call_count == 2
    assert mock_trial.state["status"] == "completed"
    assert mock_trial.state["epochs_completed"] == 20
    mock_trial.saveState.assert_called()


@patch("experiments.trainer.torch.save")
@patch("experiments.trainer.saveStates")
def test_Trainer_runHandlesPartialSavepointRemainder(
    mock_save_states, mock_torch_save, mock_trial, mock_constructor
):
    mock_trial.state["max_epoch"] = (
        15  # 15 total epochs is not divisible by savepoint = 10
    )
    trainFn = MagicMock()

    trainer = Trainer(mock_trial, trainFn, mock_constructor)
    trainer.run()

    assert trainFn.call_count == 2
    firstCallEpochs = trainFn.call_args_list[0].kwargs["epochs"]
    secondCallEpochs = trainFn.call_args_list[1].kwargs["epochs"]

    assert firstCallEpochs == 10
    assert secondCallEpochs == 5
