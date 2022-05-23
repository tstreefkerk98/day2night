import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 2
LEARNING_RATE_GEN = 2e-4
LEARNING_RATE_DISC = 2e-4
LEARNING_RATE_DECAY = None
LAMBDA_CYCLE = 10
LAMBDA_CYCLE_W = 50
LAMBDA_GRADIENT_PENALTY = 10
CLAMP_W = None
NUM_WORKERS = 4
NUM_EPOCHS = 20
CURRENT_EPOCH = 0
SAVE_MODEL = True
LOAD_MODEL = True
LOG_TRAINING = True
USE_ARCHITECTURE_B = False
CHECKPOINT_GEN_N = "genn.pth.tar"
CHECKPOINT_GEN_D = "gend.pth.tar"
CHECKPOINT_CRITIC_N = "criticn.pth.tar"
CHECKPOINT_CRITIC_D = "criticd.pth.tar"

transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
    ToTensorV2()
],
    additional_targets={"image0": "image"}
)
