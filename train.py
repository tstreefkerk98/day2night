import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
import sys
import utils
import json
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator as Generator

train_output_files = {}


def train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, base_path,
             use_ciconv, epoch):
    N_reals = 0
    N_fakes = 0

    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators N and D
        with torch.cuda.amp.autocast(enabled=False):
            fake_night = gen_N(day)
            D_N_real = disc_N(night)
            D_N_fake = disc_N(fake_night.detach())
            N_reals += D_N_real.mean().item()
            N_fakes += D_N_fake.mean().item()
            D_N_real_loss = mse(D_N_real, torch.ones_like(D_N_real))
            D_N_fake_loss = mse(D_N_fake, torch.zeros_like(D_N_fake))
            D_N_loss = D_N_real_loss + D_N_fake_loss

            fake_day = gen_D(night)
            D_D_real = disc_D(day)
            D_D_fake = disc_D(fake_day.detach())
            D_D_real_loss = mse(D_D_real, torch.ones_like(D_D_real))
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

        if idx % 200 == 0 or idx + 1 == config.BATCH_SIZE:
            save_image(fake_day * 0.5 + 0.5, f"{base_path}/saved_images_{base_path}/day_{epoch}_{idx}.png")
            save_image(fake_night * 0.5 + 0.5, f"{base_path}/saved_images_{base_path}/night_{epoch}_{idx}.png")

        save_train_output_values(idx + 1, epoch,
                                 D_N_real_loss, D_N_fake_loss, D_D_real_loss, D_D_fake_loss,
                                 loss_G_D, loss_G_N, cycle_day_loss, cycle_night_loss,
                                 disc_N.ciconv.scale.item() if use_ciconv else None,
                                 disc_D.ciconv.scale.item() if use_ciconv else None)


def save_train_output_values(batch, epoch,
                             D_N_real_loss, D_N_fake_loss, D_D_real_loss, D_D_fake_loss,
                             loss_G_D, loss_G_N, loss_C_N, loss_C_D,
                             disc_N_scale, disc_D_scale):
    size = 10
    D_N_real_loss_mean, D_N_fake_loss_mean, D_D_real_loss_mean, D_D_fake_loss_mean, \
    loss_G_D_mean, loss_G_N_mean, loss_C_N_mean, loss_C_D_mean = \
        map(lambda x: x.mean().item(),
            [D_N_real_loss, D_N_fake_loss, D_D_real_loss, D_D_fake_loss, loss_G_D, loss_G_N, loss_C_N, loss_C_D])

    train_output_obj = {
        'batch': batch,
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

    train_output_files[f"epoch_{epoch}"] = train_output_obj

    rounded_output_obj = {k: utils.format_value(v, size) for k, v in train_output_obj.items()}
    rounded_output_obj["batch"] = utils.format_value(rounded_output_obj["batch"], config.BATCH_SIZE)
    print(rounded_output_obj)


def main(use_ciconv):
    print(
        f"Training started at {utils.get_date_time(utils.get_time())}, {'' if use_ciconv else 'not'} using CIConv\n"
        "with settings:\n"
        f"BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"LEARNING_RATE: {config.LEARNING_RATE}\n"
        f"LAMBDA_CYCLE: {config.LAMBDA_CYCLE}\n"
        f"NUM_WORKERS: {config.NUM_WORKERS}\n"
        f"NUM_EPOCHS: {config.NUM_EPOCHS}\n"
        f"SAVE_MODEL: {config.SAVE_MODEL}\n"
    )

    disc_N = Discriminator(in_channels=3, use_ciconv=use_ciconv).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3, use_ciconv=use_ciconv).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_N.parameters()) + list(disc_D.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    base_path = "ciconv/" if use_ciconv else "no_ciconv/"
    checkpoint_files = [config.CHECKPOINT_GEN_N, config.CHECKPOINT_GEN_D, config.CHECKPOINT_CRITIC_N,
                        config.CHECKPOINT_CRITIC_D]
    models = [gen_N, gen_D, disc_N, disc_D]
    optimizers = [opt_gen, opt_gen, opt_disc, opt_disc]

    if len(os.listdir(base_path + "checkpoints")) != 0:
        for i in range(len(checkpoint_files)):
            load_checkpoint(
                base_path + "checkpoints/" + checkpoint_files[i],
                models[i],
                optimizers[i],
                config.LEARNING_RATE
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

    for epoch in range(config.NUM_EPOCHS):
        time = utils.get_time()
        progress = f"{epoch + 1}/{config.NUM_EPOCHS}"
        print(f"Epoch: {progress}, batch size: {len(loader)}, start time: {utils.get_date_time(time)}")

        train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, base_path,
                 use_ciconv, epoch + 1)

        utils.print_duration(utils.get_time() - time, "Epoch", progress)

        if config.SAVE_MODEL:
            for i in range(len(checkpoint_files)):
                save_checkpoint(models[i], optimizers[i], filename=base_path + "checkpoints/" + checkpoint_files[i])

        output_file_path = \
            f"{'ciconv' if use_ciconv else 'no_ciconv'}/training_outputs/training_output_{train_output_path_ext}.json"
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(train_output_files, output_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    disc_uses_ciconv = sys.argv[1]
    assert disc_uses_ciconv.lower() in ["true", "false"]

    if len(sys.argv) >= 3:
        learning_rate = eval(sys.argv[2])
        assert isinstance(learning_rate, float)
        config.LEARNING_RATE = learning_rate

    if len(sys.argv) >= 4:
        train_output_path_ext = sys.argv[3]
        assert isinstance(train_output_path_ext, str)

    main(eval(disc_uses_ciconv.title()))
