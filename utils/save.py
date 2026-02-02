import torch
import os

from pathlib import Path

from .config import SAVEDMODELSDIR

def deleteFile(fileName: str, fileDir: str):
    filePath = os.path.join(fileDir, fileName)
    try:
        os.remove(filePath)
        print(f"File deleted: {filePath}")
    except FileNotFoundError:
        print(f"File not found: {filePath}")
    except Exception as e:
        print(f"Error deleting {filePath}: {e}")

def deleteStates(stateName: str, stateDir: str=SAVEDMODELSDIR):
    deleteFile(fileName=stateName, fileDir=stateDir)

def saveTorchObject(obj, targetDir: str, fileName: str):
    targetDirPath = Path(targetDir)
    targetDirPath.mkdir(parents=True, exist_ok=True)
    
    assert fileName.endswith(".pth") or fileName.endswith(".pt"), "fileName must end with '.pt' or '.pth'."
    fileSavePath = targetDirPath / fileName

    print(f"Saving torch object to: {fileSavePath}")
    torch.save(obj=obj,
               f=fileSavePath)

def loadTorchObject(targetDir: str, fileName: str, device: str="cpu"):
    targetDirPath = Path(targetDir)
    fileLoadPath = targetDirPath / fileName

    return torch.load(fileLoadPath, map_location=device)

def loadResultsMap(resultsName: str, resultsDir: str, device: str="cpu"):
    print(f"Trying to load: {resultsDir}/{resultsName}")
    try:
        result = loadTorchObject(targetDir=resultsDir, fileName=resultsName, device=device)
        return result
    except:
        return None

def saveStates(stateName: str, stateDir: str=SAVEDMODELSDIR, **kwargs):
    state = {}
    for k, v in kwargs.items():
        state[k] = v if isinstance(v, dict) else v.state_dict()
    saveTorchObject(state, targetDir=stateDir, fileName=stateName)

def loadStates(stateName: str, stateDir: str=SAVEDMODELSDIR, **kwargs):
    try:
        state = loadTorchObject(targetDir=stateDir, fileName=stateName, device="cpu")
    except:
        return None
    
    for k, v in kwargs.items():
        if k not in state.keys() or v is None:
            continue
        v.load_state_dict(state[k])
    
    return state