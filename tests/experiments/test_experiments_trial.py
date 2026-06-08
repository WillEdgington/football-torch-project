import io
import json
from pathlib import Path
from typing import Dict
from unittest.mock import patch

import pytest

from experiments.trial import Trial


@pytest.fixture
def mock_fs():
    virtualFiles: Dict[str, str] = {}

    def mockOpenImplementation(filePath, mode: str = "r", *args, **kwargs):
        pathStr = str(filePath)

        if "w" in mode:
            stringIO = io.StringIO()
            originalClose = stringIO.close

            def closeWrapper():
                virtualFiles[pathStr] = stringIO.getvalue()
                originalClose()

            stringIO.close = closeWrapper
            return stringIO

        if pathStr not in virtualFiles:
            raise FileNotFoundError(f"No such file or directory: '{pathStr}'")
        return io.StringIO(virtualFiles[pathStr])

    with (
        patch("builtins.open", side_effect=mockOpenImplementation),
        patch("pathlib.Path.mkdir"),
        patch(
            "pathlib.Path.exists",
            autospec=True,
            side_effect=lambda self: str(self) in virtualFiles,
        ),
    ):
        yield virtualFiles


def test_Trial_initCreatesDirectories(mock_fs):
    with patch("pathlib.Path.mkdir") as mock_mkdir:
        basePath = Path("/mock/root/trial_123")
        trial = Trial(basePath)

        assert trial.definitionPath == basePath / "definition.json"
        assert trial.statePath == basePath / "state.json"
        assert mock_mkdir.call_count == 2


@patch("experiments.trial.time.time", return_value=1700000000.0)
def test_Trial_createFlow(mock_time, mock_fs):
    definition = {"train": {"epochs": 50}, "model": "ViT"}
    rootDir = "/mock/root"

    trial = Trial.create(definition, root=rootDir)

    expectedTrialId = "trial_1700000000000000"
    expectedBase = f"{rootDir}/{expectedTrialId}"

    assert str(trial.path) == expectedBase
    assert f"{expectedBase}/definition.json" in mock_fs
    assert f"{expectedBase}/state.json" in mock_fs

    savedState = json.loads(mock_fs[f"{expectedBase}/state.json"])
    assert savedState["status"] == "created"
    assert savedState["max_epoch"] == 50


def test_Trial_loadSuccess(mock_fs):
    trialDir = "/mock/root/trial_999"

    mock_fs[f"{trialDir}/definition.json"] = json.dumps({"model": "ViT"})
    mock_fs[f"{trialDir}/state.json"] = json.dumps(
        {"status": "running", "epochs_completed": 5}
    )
    trial = Trial.load(Path(trialDir))

    assert trial.getDefinition()["model"] == "ViT"
    assert trial.getState()["status"] == "running"


def test_Trial_loadMissingFilesRaisesError(mock_fs):
    trialDir = Path("/mock/root/trial_empty")

    with pytest.raises(FileNotFoundError):
        Trial.load(trialDir)


@pytest.mark.parametrize(
    "status_str, expected_bool",
    [("completed", True), ("running", False), ("created", False)],
)
def test_Trial_isComplete(mock_fs, status_str, expected_bool):
    trial = Trial(Path("/mock/trial"))
    trial.state = {"status": status_str}
    assert trial.isComplete() is expected_bool


def test_Trial_statStateUpdatesTimestamp(mock_fs):
    trialDir = "/mock/trial"
    mock_fs[f"{trialDir}/state.json"] = json.dumps({"status": "running"})

    trial = Trial(Path(trialDir))
    trial.state = {"status": "completed", "epochs_completed": 10}

    frozenTime = 1800000000.0
    with patch("experiments.trial.time.time", return_value=frozenTime):
        trial.saveState()

    updatedFileContents = json.loads(mock_fs[f"{trialDir}/state.json"])
    assert updatedFileContents["status"] == "completed"
    assert updatedFileContents["updated_at"] == frozenTime


def test_Trial_saveStateUnloadedRaisesError(mock_fs):
    trial = Trial(Path("/mock/trial"))
    trial.state = None

    with pytest.raises(RuntimeError) as exc_info:
        trial.saveState()
    assert "Trial state not loaded" in str(exc_info.value)
