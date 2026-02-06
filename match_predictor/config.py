import torch

SAVEDMODELSDIR = "saved_models"

BIGFIVELEAGUES = [
    "premier league",
    "la liga",
    "bundesliga",
    "serie a",
    "ligue 1"
]

BASEDEFINITION = {
    "model": {
        "model": "MatchPredictorV0",
        "latentSize": 20,
        "embDim": 1,
        "encoderNumDownBlocks": 2,
        "encoderAttnBlocksPerDown": 1,
        "encoderNumAttnHeads": 2,
        "encoderAttnDropout": 0.3,
        "encoderResAttn": "true",
        "encoderConvBlocksPerDown": 1,
        "encoderConvKernelSize": 3,
        "encoderConvNorm": "true",
        "encoderConvActivation": "SiLU",
        "encoderResConv": "true",
        "featExtractorDepth": 2,
        "featExtractorUseAttn": "true",
        "featExtractorResAttn": "true",
        "featExtractorAttnDropout": 0.3,
        "featExtractorNumAttnHeads": 2,
        "featExtractorUseFFN": "true",
        "featExtractorResFFN": "true",
        "featExtractorExpansionFFN": 2,
        "featExtractorLnormFFN": "true",
        "featExtractorActivationFFN": "SiLU",
        "activationMLP": "SiLU",
    },
    "optimizer": {
        "optimizer": "AdamW",
        "lr": 0.0001,
        "weight_decay": 0.0001,
    },
    "data": {
        "transform": {
            "RandomTokenUNK": {
                "prob": 0.7,
                "intensity": 0.9,
            },
            "TemporalDropout": {
                "prob": 0.1,
                "minKeep": 1,
            },
            "MissingValueAugment": {
                "prob": 0.1,
                "intensity": 0.05,
            },
            "ContinuousFeatureDropout": {
                "prob": 0.1,
                "intensity": 0.05,
            },
        },
        "batchSize": 32,
        "seqLen": 20,
        "groups": None,
    },
    "lossFn": {
        "lossFn": "CrossEntropyLoss",
        "label_smoothing": 0.0,
    },
    "train": {
        "epochs": 20,
        "savepoint": 10,
        "calcAccuracy": "true",
        "enableAmp": "true",
        "gradClipping": 1.0,
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
}

SWEEP = {
    ("model", "latentSize"): ["default", 10, 40],
    ("model", "embDim"): ["default", 2, 3],
    ("model", "encoderNumDownBlocks"): ["default", 1, 3],
    ("model", "encoderAttnBlocksPerDown"): ["default", 2],
    ("model", "encoderAttnDropout"): ["default", 0.0, 0.1],
    ("model", "encoderConvActivation"): ["default", "LeakyReLU"],
    ("model", "featExtractorDepth"): ["default", 1, 3],
    ("model", "featExtractorAttnDropout"): ["default", 0.0, 0.1],
    ("model", "featExtractorExpansionFFN"): ["default", 4],
    ("model", "featExtractorActivationFFN"): ["default", "LeakyReLU"],
    ("model", "activationMLP"): ["default", "LeakyReLU"],
    ("optimizer", "lr"): ["default", 1e-5],
    ("data", "batchSize"): ["default", 16, 64, 128],
    ("data", "seqLen"): ["default", 10, 30, 40, 50],
    ("lossFn", "label_smoothing"): ["default", 0.1, 0.2],
    ("data", "transform"): ["default", None],
}