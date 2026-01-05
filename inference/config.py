import os
import torch

class LabelConfig:
    # Class labels indices
    REAL_IDX = 0
    FAKE_IDX = 1

    # Class labels names
    REAL_NAME = 'Real'
    FAKE_NAME = 'Fake'

    # Label mappings
    ID2LABELS = {
        REAL_IDX: REAL_NAME,
        FAKE_IDX: FAKE_NAME
    }

    LABELS2ID = {
        REAL_NAME: REAL_IDX,
        FAKE_NAME: FAKE_IDX
    }


class GlobalConfig:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 240
    CROP_SIZE = 256
    RANDOM_SEED = 42
    NUM_WORKERS = 2
    BATCH_SIZE = 16