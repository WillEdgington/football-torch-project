import torch
import numpy as np
import json
import hashlib

from typing import Any, Dict

def toJSONSafe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: toJSONSafe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [toJSONSafe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [toJSONSafe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
    
def hashDefinition(definition: Dict[str, Any]) -> str:
    blob = json.dumps(definition, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()