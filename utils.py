import math

import numpy as np
import os
import random
import torch
import wandb
from PIL import Image
from torchvision.utils import save_image, make_grid

import config
import datetime
from pathlib import Path


def save_checkpoint(model, optimizer, epoch, filename=Path("my_checkpoint.pth.tar")):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    config.CURRENT_EPOCH = checkpoint['epoch'] + 1

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_images(idx, loader, epoch, fake_day, fake_night, base_path, train_output_path_tail):
    if idx % math.ceil(len(loader) / 5) == 0 or idx + 1 == len(loader):
        current_epoch = config.CURRENT_EPOCH + epoch
        save_image(fake_day * 0.5 + 0.5,
                   f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_day_{current_epoch}_{idx}.png")
        save_image(fake_night * 0.5 + 0.5,
                   f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_night_{current_epoch}_{idx}.png")


def get_images(train_folder, resize):
    return [Path(os.path.join(root, name)).resolve()
            for root, _, files in os.walk(config.TRAIN_DIR + ("_original/" if resize else "/") + train_folder)
            for name in files]


def get_wandb_img(tensor):
    grid = make_grid(tensor * 0.5 + 0.5)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    pil_img = Image.fromarray(ndarr)
    return wandb.Image(pil_img)


def dir_contains_checkpoint_files(base_path, checkpoint_files):
    return all([os.path.exists(base_path + "checkpoints/" + checkpoint_file) for checkpoint_file in checkpoint_files])


def get_time():
    return datetime.datetime.now()


def get_date_time(time):
    return time.strftime("%m/%d/%Y, %H:%M:%S")


def print_duration(difference, task, progress):
    duration_in_s = difference.total_seconds()
    hours = divmod(duration_in_s, 3600)
    minutes = divmod(hours[1], 60)
    seconds = divmod(minutes[1], 1)
    print(f"{task}: {progress} took: {hours[0]} hours, {minutes[0]} minutes, {seconds[0]} seconds")


def format_value(val, size):
    val = str(val)
    if len(val) > size:
        if "e" in val:
            e_idx = val.index("e")
            e_power = val[e_idx:]
            return val[0:(size - len(e_power))] + e_power
        return val[0:size]
    elif len(val) < size:
        return val.ljust(size, " ")
    return val


def prob(p):
    return random.random() <= p


def log_training_statistics(
        use_ciconv_d, use_ciconv_g,
        # Generators and Discriminators
        gen_D, gen_N, disc_D, disc_N,
        # Discriminator losses
        D_D_real_pred, D_D_fake_pred, D_N_real_pred, D_N_fake_pred,
        # Generator losses
        G_D_loss=None, G_N_loss=None, G_D_cycle_loss=None, G_N_cycle_loss=None, G_loss=None,
        # CycleWGAN-gp specific losses
        D_D_gradient_penalty=None, D_N_gradient_penalty=None, D_D_loss=None, D_N_loss=None,
        # CycleGAN specific losses
        D_D_real_loss=None, D_D_fake_loss=None, D_N_real_loss=None, D_N_fake_loss=None, D_loss=None
):
    log_obj = {
        "Discriminator day real prediction": D_D_real_pred,
        "Discriminator day fake prediction": D_D_fake_pred,
        "Discriminator night real prediction": D_N_real_pred,
        "Discriminator night fake prediction": D_N_fake_pred,
        "Generator day last gradient": gen_D.last.weight.grad.mean().item(),
        "Generator night last gradient": gen_N.last.weight.grad.mean().item(),
        "Generator day last gradient abs": torch.abs(gen_D.last.weight.grad).mean().item(),
        "Generator night last gradient abs": torch.abs(gen_N.last.weight.grad).mean().item(),
    }

    if all([G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss, G_loss]):
        log_obj["Generator day loss"] = G_D_loss.mean().item()
        log_obj["Generator night loss"] = G_N_loss.mean().item()
        log_obj["Generator day cycle loss"] = G_D_cycle_loss.mean().item()
        log_obj["Generator night cycle loss"] = G_N_cycle_loss.mean().item()
        log_obj["Generators total loss"] = G_loss

    if use_ciconv_d:
        log_obj["Discriminator night CIConv scale"] = disc_N.ciconv.scale.item()
        log_obj["Discriminator day CIConv scale"] = disc_D.ciconv.scale.item()
    elif use_ciconv_g:
        log_obj["Discriminator night CIConv scale"] = gen_N.ciconv.scale.item()
        log_obj["Discriminator day CIConv scale"] = gen_D.ciconv.scale.item()

    # CycleWGAN-gp specific losses
    if all([D_D_gradient_penalty, D_N_gradient_penalty, D_D_loss, D_N_loss]):
        log_obj["Discriminator day gradient penalty"] = D_D_gradient_penalty
        log_obj["Discriminator night gradient penalty"] = D_N_gradient_penalty
        log_obj["Discriminator day loss"] = D_D_loss
        log_obj["Discriminator night loss"] = D_N_loss
    # CycleGAN specific losses
    elif all([D_D_real_loss, D_D_fake_loss, D_N_real_loss, D_N_fake_loss, D_loss]):
        log_obj["Discriminator day real loss"] = D_D_real_loss.mean().item(),
        log_obj["Discriminator day fake loss"] = D_D_fake_loss.mean().item(),
        log_obj["Discriminator night real loss"] = D_N_real_loss.mean().item(),
        log_obj["Discriminator night fake loss"] = D_N_fake_loss.mean().item(),
        log_obj["Discriminators combined loss"] = D_loss,

    wandb.log(log_obj)