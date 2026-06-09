import pytest

from experiments.trial_scheduler import GridTrialScheduler, TrialScheduler, deepSet


@pytest.fixture
def mock_gridts_config():
    baseDefinition = {
        "optimizer": {"lr": 0.001},
        "model": {"latentSize": 20},
        "batchSize": 32,
    }
    sweep = {
        ("optimizer", "lr"): ["default"],
        ("model", "latentSize"): ["default", 40],
        ("batchSize",): [64, 128],
    }
    return baseDefinition, sweep


def test_TrialScheduler_nextRaisesNotImplementedError():
    scheduler = TrialScheduler()
    with pytest.raises(NotImplementedError) as exc_info:
        scheduler.next()
    assert "TrialScheduler.next must be implemented" in str(
        exc_info.value
    ), "NotImplementedError should be raised in TrialScheduler.next()"


def test_deepSet_updatesValue():
    targetDict = {"model": {"config": {"latentSize": 20}}}
    deepSet(targetDict, ("model", "config", "latentSize"), 40)
    assert (
        targetDict["model"]["config"]["latentSize"] == 40
    ), "deepSet should update nested value correctly"


def test_deepSet_ignoresDefault():
    targetDict = {"model": {"latentSize": 20}}
    deepSet(targetDict, ("model", "latentSize"), "default")
    assert (
        targetDict["model"]["latentSize"] == 20
    ), "value = 'default' should not update nested dict"


def test_GridTrialScheduler_nextProducesExpectedDefinitions(mock_gridts_config):
    base, sweep = mock_gridts_config
    scheduler = GridTrialScheduler(base, sweep)

    expectedDefinitions = [
        {"optimizer": {"lr": 0.001}, "model": {"latentSize": 20}, "batchSize": 64},
        {"optimizer": {"lr": 0.001}, "model": {"latentSize": 20}, "batchSize": 128},
        {"optimizer": {"lr": 0.001}, "model": {"latentSize": 40}, "batchSize": 64},
        {"optimizer": {"lr": 0.001}, "model": {"latentSize": 40}, "batchSize": 128},
    ]

    count = 0
    for expected in expectedDefinitions:
        count += int(scheduler.next() == expected)

    assert count == len(
        expectedDefinitions
    ), "Trial definitions from GridTrialScheduler.next() were not as expected"
    assert all(
        scheduler.next() is None for _ in range(5)
    ), "Exhausted GridTrialScheduler.next() should consistently return None"


def test_GridTrialScheduler_nextOutputIsImmutable(mock_gridts_config):
    base, sweep = mock_gridts_config
    scheduler = GridTrialScheduler(base, sweep)

    first = scheduler.next()
    first["model"]["latentSize"] = 999
    second = scheduler.next()
    assert second["model"]["latentSize"] == 20, (
        "Mutation of GridTrialScheduler.next() previous output"
        "should not affect the output returned on future calls"
    )
