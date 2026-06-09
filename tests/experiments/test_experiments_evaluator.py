import json
from unittest.mock import MagicMock, patch

import pytest

from experiments.evaluator import Evaluator


@pytest.fixture
def mock_eval_setup():
    return {
        "definition": {
            "data": {
                "transform": None,
                "batchSize": 64,
                "groups": [
                    "premier league",
                    "la liga",
                    "bundesliga",
                    "serie a",
                    "ligue 1",
                ],
            },
            "lossFn": {"lossFn": "CrossEntropyLoss", "label_smoothing": 0.0},
        },
        "hash": "hash123",
    }


@patch("experiments.evaluator.hashDefinition")
def test_Evaluator_initGeneratesHash(mock_hash, mock_eval_setup):
    mock_hash.return_value = mock_eval_setup["hash"]

    evaluator = Evaluator(
        eval=MagicMock(),
        constructor=MagicMock(),
        evalDefinition=mock_eval_setup["definition"],
    )

    assert evaluator.evalHash == mock_eval_setup["hash"]
    mock_hash.assert_called_once_with(mock_eval_setup["definition"])


@patch("experiments.evaluator.hashDefinition")
def test_Evaluator_runOrchestratesEvaluationPerDataloader(mock_hash, mock_eval_setup):
    mock_hash.return_value = mock_eval_setup["hash"]

    evalFn = MagicMock(return_value={"score": 0.95})
    evaluator = Evaluator(
        eval=evalFn,
        constructor=MagicMock(),
        evalDefinition=mock_eval_setup["definition"],
    )

    evaluator.device = "cpu"
    evaluator._loadTrial = MagicMock(
        return_value={
            "dataloaders": {"train_split": "loader_a", "val_split": "loader_b"},
            "model": "opaque_model_object",
        }
    )

    trial = MagicMock()
    trial._definition = "active_cache_payload"

    evaluator.save = MagicMock(return_value=mock_eval_setup["hash"])
    resultingHash = evaluator.run(trial)

    assert resultingHash == mock_eval_setup["hash"]
    assert evalFn.call_count == 2
    assert trial._definition is None

    evaluator.save.assert_called_once_with(
        trial, {"train_split": {"score": 0.95}, "val_split": {"score": 0.95}}
    )


@patch("experiments.evaluator.hashDefinition")
def test_Evaluator_saveBypassesDiskWriteIfFileAlreadyExists(
    mock_hash, mock_eval_setup, tmp_path
):
    mock_hash.return_value = mock_eval_setup["hash"]
    evaluator = Evaluator(
        eval=MagicMock(),
        constructor=MagicMock(),
        evalDefinition=mock_eval_setup["definition"],
    )

    trial = MagicMock()
    trial.evalsPath = tmp_path / "evals"
    trial.evalsPath.mkdir()

    existingFile = trial.evalsPath / f"eval_{mock_eval_setup['hash']}.json"
    existingFile.write_text("pre_existing_encoded_results")

    returnedHash = evaluator.save(trial, {"new_split": {"score": 0.0}})

    assert returnedHash == mock_eval_setup["hash"]
    assert existingFile.read_text() == "pre_existing_encoded_results"


@patch("experiments.evaluator.hashDefinition")
@patch("experiments.evaluator.toJSONSafe", side_effect=lambda x: x)
def test_Evaluator_saveWritesEvaluationPayloadAtomically(
    mock_json_safe, mock_hash, mock_eval_setup, tmp_path
):
    mock_hash.return_value = mock_eval_setup["hash"]
    evaluator = Evaluator(
        eval=MagicMock(),
        constructor=MagicMock(),
        evalDefinition=mock_eval_setup["definition"],
    )

    trial = MagicMock()
    trial.evalsPath = tmp_path / "evals"
    trial.evalsPath.mkdir()

    freshResults = {"test_split": {"loss": 0.04}}
    returnedHash = evaluator.save(trial, freshResults)

    assert returnedHash == mock_eval_setup["hash"]

    targetOutputFile = trial.evalsPath / f"eval_{mock_eval_setup['hash']}.json"
    assert targetOutputFile.exists()
    assert not (trial.evalsPath / f"eval_{mock_eval_setup['hash']}.tmp").exists()

    with open(targetOutputFile, "r") as f:
        storedPayload = json.load(f)

    assert storedPayload["eval_hash"] == mock_eval_setup["hash"]
    assert storedPayload["eval_definition"] == mock_eval_setup["definition"]
    assert storedPayload["results"] == freshResults
