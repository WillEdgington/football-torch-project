from unittest.mock import MagicMock, mock_open, patch

import pytest

from experiments.trial_result import TrialResult


@pytest.fixture
def mock_trial():
    trial = MagicMock()
    trial.getDefinition.return_value = {"model": "MatchPredictorV0", "seed": 42}

    trial.metricsPath = MagicMock()
    trial.evalsPath = MagicMock()
    return trial


@pytest.fixture
def sample_eval_data():
    return {
        "eval_hash": "hash123",
        "results": {
            "train": {"loss": 0.25, "accuracy": 0.88},
            "val": {"loss": 0.31, "accuracy": 0.82},
        },
    }


@pytest.fixture
def mock_json():
    jsonFile = MagicMock()
    jsonFile.suffix = ".json"
    return jsonFile


@pytest.fixture
def mock_txt():
    txtFile = MagicMock()
    txtFile.suffix = ".txt"
    return txtFile


def test_TrialResult_definitionProperty(mock_trial):
    result = TrialResult(mock_trial)

    assert result.definition == {"model": "MatchPredictorV0", "seed": 42}
    mock_trial.getDefinition.assert_called_once()


def test_TrialResult_metricsReturnsNoneIfFileMissing(mock_trial):
    mock_trial.metricsPath.exists.return_value = False
    result = TrialResult(mock_trial)

    assert result.metrics is None


@patch("torch.load")
def test_TrialResult_metricsLoadSuccessfully(mock_torch_load, mock_trial):
    mock_trial.metricsPath.exists.return_value = True
    dummyMetrics = {"epoch_loss": [0.5, 0.3, 0.1]}
    mock_torch_load.return_value = dummyMetrics

    result = TrialResult(mock_trial)

    assert result.metrics == dummyMetrics
    assert result.metrics == dummyMetrics  # verify caching logic
    mock_torch_load.assert_called_once_with(mock_trial.metricsPath)


def test_TrialResult_evalsParsingAndExtensionSkipping(
    mock_trial, sample_eval_data, mock_json, mock_txt
):
    mock_trial.evalsPath.iterdir.return_value = [mock_json, mock_txt]

    result = TrialResult(mock_trial)

    with (
        patch("builtins.open", mock_open()),
        patch("json.load", return_value=sample_eval_data),
    ):
        parsedEvals = result.evals

        assert "hash123" in parsedEvals
        assert result.evalHashes() == ["hash123"]
        assert len(parsedEvals) == 1


def test_TrialResult_getEvalRaisesValueErrorOnMissingHash(mock_trial):
    mock_trial.evalsPath.iterdir.return_value = []
    result = TrialResult(mock_trial)

    with pytest.raises(ValueError) as exc_info:
        result.getEval("invalid_hash")
    assert "Eval hash not found: invalid_hash" in str(exc_info.value)


def test_TrialResult_getSplitSuccessAndFailures(
    mock_trial, sample_eval_data, mock_json
):
    mock_trial.evalsPath.iterdir.return_value = [mock_json]
    result = TrialResult(mock_trial)

    with (
        patch("builtins.open", mock_open()),
        patch("json.load", return_value=sample_eval_data),
    ):
        assert result.splits("hash123") == ["train", "val"]
        valSplit = result.getSplit("hash123", "val")
        assert valSplit["accuracy"] == 0.82

        with pytest.raises(ValueError) as exc_info:
            result.getSplit("hash123", "invalid_split")
        assert "Split not found: invalid_split" in str(exc_info.value)
