import numpy as np
import os
import random
import torch
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
    print(f"=> Current epoch: {checkpoint['epoch'] + 1}")

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


def get_images(train_folder, resize):
    return [Path(os.path.join(root, name)).resolve()
            for root, _, files in os.walk(config.TRAIN_DIR + ("_original/" if resize else "/") + train_folder)
            for name in files]


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