import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
# LEARNING_RATE = 1e-5
LEARNING_RATE_G = 1e-5
LEARNING_RATE_D = 2.5e-6
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 2
SAVE_MODEL = False
LOAD_MODEL = False
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
