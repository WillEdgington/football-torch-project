from copy import deepcopy
from itertools import product
from typing import Dict, List, Any, Tuple, Iterator

class TrialScheduler:
    def next(self) -> Dict[str, Any]|None:
        raise NotImplementedError("TrialScheduler.next must be implemented")

def deepSet(d: Dict[str, Any], path: Tuple[str, ...], value: Any):
    if value == "default":
        return
    cur = d
    for key in path[:-1]:
        cur = cur[key]
    cur[path[-1]] = value

class GridTrialScheduler(TrialScheduler):
    def __init__(self,
                 baseDefinition: Dict[str, Any],
                 sweep: Dict[Tuple[str, ...], List[Any]]):
        self.baseDefinition = baseDefinition
        self.paths = list(sweep.keys())
        self.values = list(sweep.values())

        self._grid: Iterator[Tuple[Any, ...]] = product(*self.values)

    def next(self) -> Dict[str, Any]|None:
        try:
            combo = next(self._grid)
        except StopIteration:
            return None
    
        definition = deepcopy(self.baseDefinition)
        for path, value in zip(self.paths, combo):
            deepSet(definition, path, value)

        return definition