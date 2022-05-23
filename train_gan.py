import math
import torch
import config
from utils import save_images, prob, log_training_statistics


def get_disc_cycle_gan_losses(disc, mse, real, fake):
    with torch.cuda.amp.autocast(enabled=False):
        # Flip 5% of training labels going into Discriminator
        if prob(0.05):
            disc_real_pred = disc(fake.detach())
            disc_fake_pred = disc(real)
        else:
            disc_real_pred = disc(real)
            disc_fake_pred = disc(fake.detach())

        disc_real_pred_mean = disc_real_pred.mean().item()
        disc_fake_pred_mean = disc_fake_pred.mean().item()

        disc_real_loss = mse(disc_real_pred, torch.ones_like(disc_real_pred))
        # One sided label smoothing Discriminator
        if disc_real_loss.mean().item() < 0.1:
            disc_real_loss = mse(disc_real_pred, torch.full_like(disc_real_pred, 0.9))
        disc_fake_loss = mse(disc_fake_pred, torch.zeros_like(disc_fake_pred))

        # Add losses together
        disc_loss = disc_real_loss + disc_fake_loss

        return disc_real_pred_mean, disc_fake_pred_mean, disc_real_loss, disc_fake_loss, disc_loss


def train_gen_cycle_gan(disc_D, disc_N, gen_D, gen_N, opt_gen, g_scaler, mse, l1, day, fake_day, night, fake_night):
    # Train Generators N and D
    with torch.cuda.amp.autocast(enabled=False):
        # Adversarial loss for both generators
        D_N_fake = disc_N(fake_night)
        D_D_fake = disc_D(fake_day)
        G_N_loss = mse(D_N_fake, torch.ones_like(D_N_fake))
        G_D_loss = mse(D_D_fake, torch.ones_like(D_D_fake))

        # Cycle loss
        cycle_day = gen_D(fake_night)
        cycle_night = gen_N(fake_day)
        G_D_cycle_loss = l1(day, cycle_day) * config.LAMBDA_CYCLE
        G_N_cycle_loss = l1(night, cycle_night) * config.LAMBDA_CYCLE

        # Add all together
        G_loss = (
                G_D_loss
                + G_N_loss
                + G_D_cycle_loss
                + G_N_cycle_loss
        )

    opt_gen.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(opt_gen)
    g_scaler.update()

    return G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss, G_loss


def train_cycle_gan(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, base_path,
                    epoch, scheduler_opt_disc, scheduler_opt_gen, use_ciconv_d, use_ciconv_g, train_output_path_tail):
    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        # Train Discriminators N and D
        with torch.cuda.amp.autocast(enabled=False):
            fake_night = gen_N(day)
            fake_day = gen_D(night)

        D_N_real_pred, D_N_fake_pred, D_N_real_loss, D_N_fake_loss, D_N_loss = \
            get_disc_cycle_gan_losses(disc_N, mse, night, fake_night)

        D_D_real_pred, D_D_fake_pred, D_D_real_loss, D_D_fake_loss, D_D_loss = \
            get_disc_cycle_gan_losses(disc_D, mse, day, fake_day)

        # Put it together
        D_loss = (D_N_loss + D_D_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators N and D
        G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss, G_loss = \
            train_gen_cycle_gan(disc_D, disc_N, gen_D, gen_N, opt_gen, g_scaler, mse, l1, day, fake_day, night,
                                fake_night)

        # Save images
        if config.SAVE_MODEL:
            save_images(idx, loader, epoch, fake_day, fake_night, base_path, train_output_path_tail)

        # Log training statistics
        if config.LOG_TRAINING:
            log_training_statistics(
                use_ciconv_d, use_ciconv_g,
                # Generators and Discriminators
                gen_D, gen_N, disc_D, disc_N,
                # Discriminator losses
                D_D_real_pred, D_D_fake_pred, D_N_real_pred, D_N_fake_pred, D_D_loss, D_N_loss,
                # Generator losses
                G_D_loss=G_D_loss, G_N_loss=G_N_loss, G_D_cycle_loss=G_D_cycle_loss, G_N_cycle_loss=G_N_cycle_loss,
                G_loss=G_loss,
                # CycleGAN specific losses
                D_D_real_loss=D_D_real_loss, D_D_fake_loss=D_D_fake_loss, D_N_real_loss=D_N_real_loss,
                D_N_fake_loss=D_N_fake_loss, D_loss=D_loss
            )

        # Print progress
        if idx % math.ceil(len(loader) / 10) == 0:
            print(f"Batch {idx} out of {len(loader)} completed")

        if config.LEARNING_RATE_DECAY:
            scheduler_opt_disc.step()
            scheduler_opt_gen.step()
