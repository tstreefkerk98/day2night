import argparse
import os
import torch

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import sys

sys.path.insert(1, "../")
from generator_model import Generator
from torchvision.utils import save_image


def transform(img):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()
    ])(image=img)


def main(dir_to_convert, dir_fake, checkpoint_path, ciconv_gen):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    gen = Generator(img_channels=3, num_residuals=9, use_ciconv=ciconv_gen).to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])

    for image_path in os.listdir(dir_to_convert):
        real_img = np.array(Image.open(f"{dir_to_convert}/{image_path}").convert("RGB"))
        real_img = (transform(real_img)["image"][None, :]).to(DEVICE)
        fake_img = gen(real_img).detach()
        save_image(fake_img, f"{dir_fake}/converted_{image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_real", type=str, help="Path to the directory with real images")
    parser.add_argument("--dir_fake", type=str, help="Path to the directory to store fake images")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint")
    parser.add_argument("--ciconv_gen", action="store_true", help="Path to the checkpoint")

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments
    if args.checkpoint_path is not None:

        main(
            args.dir_real,
            args.dir_fake,
            args.checkpoint_path,
            "gen_" in args.checkpoint_path
        )
    else:
        print("No checkpoint_path was provided")
