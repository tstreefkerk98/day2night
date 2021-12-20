import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import config
import sys
import utils
import env
import json
import wandb
from PIL import Image
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from discriminator_model import Discriminator
from generator_model import Generator as Generator

train_output_files = {}


def train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, base_path,
             use_ciconv):
    N_reals = 0
    N_fakes = 0

    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators N and D
        with torch.cuda.amp.autocast(enabled=False):
            fake_night = gen_N(day)

            # flip 5% of training labels going into Discriminator
            if utils.prob(0.05):
                D_N_real = disc_N(fake_night.detach())
                D_N_fake = disc_N(night)
            else:
                D_N_real = disc_N(night)
                D_N_fake = disc_N(fake_night.detach())

            D_N_real_mean = D_N_real.mean().item()
            D_N_fake_mean = D_N_fake.mean().item()
            N_reals += D_N_real_mean
            N_fakes += D_N_fake_mean

            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            # one sided label smoothing Discriminator Night
            if D_N_real_loss.mean().item() < 0.1:
                D_N_real_loss = mse(D_N_real, torch.full_like(D_N_real, 0.9))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            fake_day = gen_D(night)

            # flip 5% of training labels going into Discriminator
            if utils.prob(0.05):
                D_D_real = disc_D(fake_day.detach())
                D_D_fake = disc_D(day)
            else:
                D_D_real = disc_D(day)
                D_D_fake = disc_D(fake_day.detach())

            D_D_real_loss = mse(D_D_real, torch.ones_like(D_D_real))
            # one sided label smoothing Discriminator Day
            if D_D_real_loss.mean().item() < 0.1:
                D_D_real_loss = mse(D_D_real, torch.full_like(D_D_real, 0.9))
            D_D_fake_loss = mse(D_D_fake, torch.zeros_like(D_D_fake))
            D_D_loss = D_D_real_loss + D_D_fake_loss

            # put it together
            D_loss = (D_N_loss + D_D_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators N and D
        with torch.cuda.amp.autocast(enabled=False):
            # adversarial loss for both generators
            D_N_fake = disc_N(fake_night)
            D_D_fake = disc_D(fake_day)
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))
            loss_G_D = mse(D_D_fake, torch.ones_like(D_D_fake))

            # cycle loss
            cycle_day = gen_D(fake_night)
            cycle_night = gen_N(fake_day)
            cycle_day_loss = l1(day, cycle_day) * config.LAMBDA_CYCLE
            cycle_night_loss = l1(night, cycle_night) * config.LAMBDA_CYCLE

            # add all together
            G_loss = (
                    loss_G_D
                    + loss_G_N
                    + cycle_day_loss
                    + cycle_night_loss
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        epoch_idx = len(train_output_files) - 1
        if idx == 0:
            epoch_idx += 1
            train_output_files[f"epoch_{epoch_idx}"] = {}

        if idx % math.ceil(len(loader) / 5) == 0 or idx + 1 == len(loader):
            save_image(fake_day * 0.5 + 0.5,
                       f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_day_{epoch_idx}_{idx}.png")
            save_image(fake_night * 0.5 + 0.5,
                       f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_night_{epoch_idx}_{idx}.png")

        D_N_real_loss_mean, D_N_fake_loss_mean, D_D_real_loss_mean, D_D_fake_loss_mean, \
        loss_G_D_mean, loss_G_N_mean, loss_C_N_mean, loss_C_D_mean = \
            map(lambda x: x.mean().item(),
                [D_N_real_loss, D_N_fake_loss, D_D_real_loss, D_D_fake_loss, loss_G_D, loss_G_N, cycle_day_loss,
                 cycle_night_loss])

        save_train_output_values(idx, len(loader), epoch_idx, D_N_real_mean, D_N_fake_mean, N_reals, N_fakes,
                                 D_N_real_loss_mean, D_N_fake_loss_mean, D_D_real_loss_mean, D_D_fake_loss_mean,
                                 loss_G_D_mean, loss_G_N_mean, loss_C_D_mean, loss_C_N_mean,
                                 disc_N.ciconv.scale.item() if use_ciconv else None,
                                 disc_D.ciconv.scale.item() if use_ciconv else None)

        log_obj = {
            "Discriminator night real prediction": D_N_real_mean,
            "Discriminator night fake prediction": D_N_fake_mean,
            "Discriminator night real loss": D_N_real_loss_mean,
            "Discriminator night fake loss": D_N_fake_loss_mean,
            "Discriminator day real loss": D_D_real_loss_mean,
            "Discriminator day fake loss": D_D_fake_loss_mean,
            "Loss generator day": loss_G_D_mean,
            "Loss generator night": loss_G_N_mean,
            "Cycle loss day": loss_C_D_mean,
            "Cycle loss night": loss_C_N_mean,
            # "Real_day": get_wandb_img(day),
            # "Fake_day": get_wandb_img(fake_day),
            # "Real_night": get_wandb_img(night),
            # "Fake_night": get_wandb_img(fake_night),
            "Epoch": epoch_idx,
            "Batch": idx
        }

        if use_ciconv:
            log_obj["Discriminator night CIConv scale"] = disc_N.ciconv.scale.item()
            log_obj["Discriminator day CIConv scale"] = disc_D.ciconv.scale.item()

        wandb.log(log_obj)


def get_wandb_img(tensor):
    grid = make_grid(tensor * 0.5 + 0.5)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    pil_img = Image.fromarray(ndarr)
    return wandb.Image(pil_img)


def save_train_output_values(batch, batches, epoch_idx, D_N_real_mean, D_N_fake_mean, N_reals, N_fakes,
                             D_N_real_loss_mean, D_N_fake_loss_mean, D_D_real_loss_mean, D_D_fake_loss_mean,
                             loss_G_D_mean, loss_G_N_mean, loss_C_N_mean, loss_C_D_mean,
                             disc_N_scale, disc_D_scale):
    size = 10
    train_output_obj = {
        'D_N_real_mean': D_N_real_mean,
        'D_N_fake_mean': D_N_fake_mean,
        'N_real_avg': N_reals / (batch + 1),
        'N_fake_avg': N_fakes / (batch + 1),
        'D_N_real_loss_mean': D_N_real_loss_mean,
        'D_N_fake_loss_mean': D_N_fake_loss_mean,
        'D_D_real_loss_mean': D_D_real_loss_mean,
        'D_D_fake_loss_mean': D_D_fake_loss_mean,
        'loss_G_D_mean': loss_G_D_mean,
        'loss_G_N_mean': loss_G_N_mean,
        'loss_C_N_mean': loss_C_N_mean,
        'loss_C_D_mean': loss_C_D_mean,
        'D_N_scale': disc_N_scale,
        'D_D_scale': disc_D_scale
    }

    train_output_files[f"epoch_{epoch_idx}"][f"batch_{batch}"] = train_output_obj

    keys_to_print = ["D_N_real_mean", "D_N_fake_mean", "N_real_avg", "N_fake_avg", "D_N_scale", "D_D_scale"]
    rounded_output_obj = {k: train_output_obj[k] for k in keys_to_print}
    rounded_output_obj = {k: utils.format_value(v, size) for k, v in rounded_output_obj.items()}
    print(f"Batch: {utils.format_value(batch, len(str(batches)))} =", rounded_output_obj)


def create_label_smoothing_tensor(input_tensor, r1, r2):
    return (r1 - r2) * torch.rand_like(input_tensor) + r2


def main():
    training_start_time = utils.get_time()
    print(
        f"Training started at {utils.get_date_time(training_start_time)}, {'' if use_ciconv else 'not '}using CIConv\n"
        "with settings:\n"
        f"BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"LEARNING_RATE_G: {config.LEARNING_RATE_G}\n"
        f"LEARNING_RATE_D: {config.LEARNING_RATE_D}\n"
        f"LAMBDA_CYCLE: {config.LAMBDA_CYCLE}\n"
        f"NUM_WORKERS: {config.NUM_WORKERS}\n"
        f"NUM_EPOCHS: {config.NUM_EPOCHS}\n"
        f"SAVE_MODEL: {config.SAVE_MODEL}\n"
        f"LOAD_MODEL: {config.LOAD_MODEL}\n"
    )

    disc_N = Discriminator(in_channels=3, use_ciconv=use_ciconv).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3, use_ciconv=use_ciconv).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_N.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE_D,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE_G,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    base_path = "ciconv/" if use_ciconv else "no_ciconv/"

    file_path = f"{base_path}training_outputs/training_output_{train_output_path_tail}.json"
    if os.path.isfile(file_path):
        with open(file_path) as output_file:
            global train_output_files
            train_output_files = json.load(output_file)

    checkpoint_files = [train_output_path_tail + "_" + config.CHECKPOINT_GEN_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_GEN_D,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_D]
    models = [gen_N, gen_D, disc_N, disc_D]
    optimizers = [opt_gen, opt_gen, opt_disc, opt_disc]
    learning_rates = [config.LEARNING_RATE_G, config.LEARNING_RATE_G, config.LEARNING_RATE_D, config.LEARNING_RATE_D]

    if config.LOAD_MODEL and dir_contains_checkpoint_files(base_path, checkpoint_files):
        for i in range(len(checkpoint_files)):
            load_checkpoint(
                base_path + "checkpoints/" + checkpoint_files[i],
                models[i],
                optimizers[i],
                learning_rates[i]
            )

    dataset = DayNightDataset(
        root_day=config.TRAIN_DIR + "/day",
        root_night=config.TRAIN_DIR + "/night",
        transform=config.transforms,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    wandb.watch(
        [gen_D, gen_N, disc_N, disc_D],
        criterion=None, log="gradients", log_freq=math.ceil(len(loader) / 5), idx=None, log_graph=False
    )

    for epoch in range(config.NUM_EPOCHS):
        time = utils.get_time()
        progress = f"{epoch + 1}/{config.NUM_EPOCHS}"
        print(f"Epoch: {progress}, batches: {len(loader)}, start time: {utils.get_date_time(time)}")

        train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, base_path,
                 use_ciconv)

        utils.print_duration(utils.get_time() - time, "Epoch", progress)

        if config.SAVE_MODEL:
            for i in range(len(checkpoint_files)):
                save_checkpoint(models[i], optimizers[i], filename=base_path + "checkpoints/" + checkpoint_files[i])

        output_file_path = \
            f"{'ciconv' if use_ciconv else 'no_ciconv'}/training_outputs/training_output_{train_output_path_tail}.json"
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(train_output_files, output_file, ensure_ascii=False, indent=4)

        if epoch + 1 == config.NUM_EPOCHS:
            utils.print_duration(utils.get_time() - training_start_time, "Training", f"{config.NUM_EPOCHS} epochs")


def dir_contains_checkpoint_files(base_path, checkpoint_files):
    return all([os.path.exists(base_path + "checkpoints/" + checkpoint_file) for checkpoint_file in checkpoint_files])


if __name__ == "__main__":
    disc_uses_ciconv = sys.argv[1]
    assert disc_uses_ciconv.lower() in ["true", "false"]

    learning_rate_g = eval(sys.argv[2])
    assert isinstance(learning_rate_g, float)
    config.LEARNING_RATE_G = learning_rate_g

    learning_rate_d = eval(sys.argv[3])
    assert isinstance(learning_rate_d, float)
    config.LEARNING_RATE_D = learning_rate_d

    batch_size = eval(sys.argv[4])
    assert isinstance(batch_size, int)
    config.BATCH_SIZE = batch_size

    num_epochs = eval(sys.argv[5])
    assert isinstance(num_epochs, int)
    config.NUM_EPOCHS = num_epochs

    train_output_path_tail = sys.argv[6]
    assert isinstance(train_output_path_tail, str)

    use_ciconv = eval(disc_uses_ciconv.title())

    wandb.login(key=env.WANDB_KEY)
    wandb.init(project="day2night", entity="tstreefkerk")
    wandb.config = {
        "ciconv": use_ciconv,
        "learning_rate_d": config.LEARNING_RATE_D,
        "learning_rate_g": config.LEARNING_RATE_G,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "file_extension": train_output_path_tail
    }

    main()
