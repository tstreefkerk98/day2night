import os
import sys
import torch
import config
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.utils import save_image
from random import choice


def classify_image():
    disc = Discriminator(use_ciconv=False).to(config.DEVICE)
    checkpoint_file = "no_ciconv/checkpoints/criticd.pth.tar"
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    disc.load_state_dict(checkpoint["state_dict"])

    img_path = "data/train/day/aachen_000000_000019_leftImg8bit.png"
    img = np.array(Image.open(img_path).convert("RGB"))
    img = transform(img)["image"][None, :]
    res = disc(img).detach()
    mean = res.mean().item()


def generate_fake_img():
    # load generator with checkpoints weights
    gen = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    base_path = "ciconv" if use_ciconv else "no_ciconv"
    checkpoint_file = f"gen{'n' if generate_fake_night else 'd'}.pth.tar"
    checkpoint = torch.load(f"{base_path}/checkpoints/{checkpoint_file}", map_location=config.DEVICE)
    gen.load_state_dict(checkpoint["state_dict"])

    # load image and convert to opposite domain
    (retrieve_path, save_path) = ("day", "day2night") if generate_fake_night else ("night", "night2day")
    retrieve_folder_path = f"data/test/{retrieve_path}"
    retrieve_folder_size = len(os.listdir(retrieve_folder_path))

    stack = []
    image_pairs = ()
    for _ in range(amount):
        rand = choice([i for i in range(retrieve_folder_size) if i not in stack])
        stack.append(rand)

        img_path = f"{retrieve_folder_path}/{retrieve_path}_{rand}.png"
        real_img = np.array(Image.open(img_path).convert("RGB"))
        real_img = transform(real_img)["image"][None, :]
        fake_img = gen(real_img).detach()
        image_pairs += (adjust_tensor(real_img), adjust_tensor(fake_img))

    # save images
    save_folder_path = f"{base_path}/generated_images/{save_path}"
    save_folder_size = len(os.listdir(save_folder_path))
    save_image(torch.stack(image_pairs), f"{save_folder_path}/{save_path}_{save_folder_size}.png")


def adjust_tensor(img):
    return torch.squeeze(img * 0.5 + 0.5)


def transform(img):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2()
    ])(image=img)


if __name__ == "__main__":
    use_ciconv = eval(sys.argv[1])
    generate_fake_night = eval(sys.argv[2])
    amount = eval(sys.argv[3])
    assert all(isinstance(b, bool) for b in [use_ciconv, generate_fake_night]) and isinstance(amount, int)
    generate_fake_img()
