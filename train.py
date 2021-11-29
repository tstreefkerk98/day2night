import torch
import torch.nn as nn
import torch.optim as optim
import config
import sys
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from no_ciconv.discriminator_model import Discriminator as Discriminator_reg
from ciconv.discriminator_model_ciconv import Discriminator as Discriminator_ciconv
from no_ciconv.generator_model import Generator as Generator


def train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, base_path):
    N_reals = 0
    N_fakes = 0

    for idx, (day, night) in enumerate(loader):
        print(f"Dataset: {idx + 1}/{len(loader.dataset)}")
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.cuda.amp.autocast():
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

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_N_fake = disc_N(fake_night)
            D_D_fake = disc_D(fake_day)
            loss_G_N = mse(D_N_fake, torch.ones_like(D_N_fake))
            loss_G_D = mse(D_D_fake, torch.ones_like(D_D_fake))

            # cycle loss
            cycle_day = gen_D(fake_night)
            cycle_night = gen_N(fake_day)
            cycle_day_loss = l1(day, cycle_day)
            cycle_night_loss = l1(night, cycle_night)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_day = gen_D(day)
            identity_night = gen_N(night)
            identity_day_loss = l1(day, identity_day)
            identity_night_loss = l1(night, identity_night)

            # add all together
            G_loss = (
                    loss_G_D
                    + loss_G_N
                    + cycle_day_loss * config.LAMBDA_CYCLE
                    + cycle_night_loss * config.LAMBDA_CYCLE
                    + identity_day_loss * config.LAMBDA_IDENTITY
                    + identity_night_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_day * 0.5 + 0.5, f"{base_path}/saved_images_{base_path}/day_{idx}.png")
            save_image(fake_night * 0.5 + 0.5, f"{base_path}/saved_images_{base_path}/night_{idx}.png")

        print(f"N_real={N_reals / (idx + 1)}", f"N_fake={N_fakes / (idx + 1)}")


def main(use_ciconv):
    Discriminator = Discriminator_ciconv if use_ciconv else Discriminator_reg
    disc_N = Discriminator(in_channels=3).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3).to(config.DEVICE)
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

    if config.LOAD_MODEL:
        for i in range(len(checkpoint_files)):
            load_checkpoint(base_path + "checkpoints/" + checkpoint_files[i], models[i], opt_gen, config.LEARNING_RATE)

    dataset = DayNightDataset(
        root_day=config.TRAIN_DIR + "/day",
        root_night=config.TRAIN_DIR + "/night",
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f"Epoch: {epoch + 1}/{config.NUM_EPOCHS}")
        train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler, base_path)

        if config.SAVE_MODEL:
            optimizers = [opt_gen, opt_gen, opt_disc, opt_disc]
            for i in range(len(checkpoint_files)):
                save_checkpoint(models[i], optimizers[i], filename=base_path + "checkpoints/" + checkpoint_files[i])


if __name__ == "__main__":
    disc_uses_ciconv = sys.argv[1]
    assert disc_uses_ciconv.lower() in ["true", "false"]
    main(eval(disc_uses_ciconv.title()))
