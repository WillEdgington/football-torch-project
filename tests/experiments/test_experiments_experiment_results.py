from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from experiments.experiment_results import (
    ExperimentResults,
    addCompositeScore,
    normaliseMetric,
)


@pytest.fixture
def simpleDf():
    return pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0, 5.0]})


@pytest.fixture
def multiColDf():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
        }
    )


def test_normaliseMetric_standard_outputMeanIsNearZero(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="standard")
    assert abs(result.mean()) < 1e-6


def test_normaliseMetric_standard_outputStdIsNearOne(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="standard")
    assert abs(result.std() - 1.0) < 1e-6


def test_normaliseMetric_minmax_outputMinIsZero(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="minmax")
    assert result.min() == pytest.approx(0.0)


def test_normaliseMetric_minmax_outputMaxIsOne(simpleDf):
    result = normaliseMetric(simpleDf, "score", method="minmax")
    assert result.max() == pytest.approx(1.0)


def test_normaliseMetric_missingColumn_raisesValueError(simpleDf):
    with pytest.raises(ValueError):
        normaliseMetric(simpleDf, "missing")


def test_addCompositeScore_mismatchedKeys_raisesValueError(multiColDf):
    with pytest.raises(ValueError):
        addCompositeScore(
            multiColDf,
            weights={"a": 1.0},
            ascending={"a": False, "b": False},
        )


def test_addCompositeScore_outputColumnPresentInDf(multiColDf):
    result = addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert "composite_score" in result.columns


def test_addCompositeScore_customColName_isPresentInDf(multiColDf):
    result = addCompositeScore(
        multiColDf,
        weights={"a": 1.0},
        ascending={"a": False},
        colName="my_score",
    )
    assert "my_score" in result.columns


def test_addCompositeScore_higherRawValue_scoresHigherWhenAscendingFalse(multiColDf):
    result = addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert result["composite_score"].iloc[2] > result["composite_score"].iloc[0]


def test_addCompositeScore_doesNotMutateInputDf(multiColDf):
    colsBefore = list(multiColDf.columns)
    addCompositeScore(multiColDf, weights={"a": 1.0}, ascending={"a": False})
    assert list(multiColDf.columns) == colsBefore


@pytest.fixture
def mock_results_engine(tmp_path):
    (tmp_path / "trials").mkdir()
    (tmp_path / "trials.json").write_text("[]")

    with patch.object(ExperimentResults, "_loadTrials") as mock_load:
        mock_load.return_value = [
            {"id": 101, "path": "/fake/path/trial_101.json", "evals": ["hash_xyz"]},
            {"id": 202, "path": "/fake/path/trial_202.json", "evals": ["hash_abc"]},
        ]
        engine = ExperimentResults(root=tmp_path)
        return engine


def test_ExperimentResults_initEnforcesDirectoryStructureExistence(tmp_path):
    with pytest.raises(AssertionError) as exc_info:
        ExperimentResults(root=tmp_path)
    assert "could not find any trials at" in str(exc_info.value)


@patch("experiments.experiment_results.Trial.load")
@patch("experiments.experiment_results.TrialResult")
def test_ExperimentResults_getTrialRoutingBoundaries(
    mock_trial_result, mock_trial_load, mock_results_engine
):
    mockLoadedTrial = MagicMock()
    mock_trial_load.return_value = mockLoadedTrial
    mock_trial_result.return_value = "wrapped_trial_result_object"

    result = mock_results_engine.getTrial(trial_id=101)
    mock_trial_load.assert_called_once_with(Path("/fake/path/trial_101.json"))
    assert result == "wrapped_trial_result_object"

    with pytest.raises(KeyError):
        mock_results_engine.getTrial(trial_id=999)


@patch("experiments.experiment_results.Trial.load")
def test_ExperimentResults_toDataFrameMatrixFilteringAndSplitting(
    mock_trial_load, mock_results_engine
):
    mock_trial = MagicMock()
    mock_trial.getDefinition.return_value = {"learning_rate": 0.01, "batch_size": 32}
    mock_trial_load.return_value = mock_trial

    mock_results_engine._loadEval = MagicMock(
        return_value={"results": {"test": {"loss": 0.15, "accuracy": 0.92}}}
    )

    df = mock_results_engine.toDataFrame(evalHash="hash_xyz", split=False)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "trial_id"
    assert 101 in df.index
    assert "learning_rate" in df.columns
    assert "test.loss" in df.columns

    defDf, evalDf = mock_results_engine.toDataFrame(evalHash="hash_xyz", split=True)
    assert isinstance(defDf, pd.DataFrame)
    assert isinstance(evalDf, pd.DataFrame)
    assert "learning_rate" in defDf.columns
    assert "test.loss" in evalDf.columns
