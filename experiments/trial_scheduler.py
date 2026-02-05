from typing import Dict, List, Any

class TrialScheduler:     
    def next(self) -> Dict[str, Any]|None:
        raise NotImplementedError("TrialScheduler.next must be implemented")