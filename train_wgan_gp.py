import math
import torch
import config
from utils import save_images, log_training_statistics
from torch.autograd import grad


def train_disc_cycle_wgan_gp(disc, real, fake, opt_disc, d_scaler):
    with torch.cuda.amp.autocast(enabled=False):
        # Predictions
        disc_real_pred = -disc(real)
        disc_fake_pred = disc(fake.detach())

        disc_real_pred_mean = torch.mean(disc_real_pred)
        disc_fake_pred_mean = torch.mean(disc_fake_pred)

        # Gradient penalty
        disc_gradients = compute_gradient_penalty(disc, real, fake)
        disc_gradient_penalty = config.LAMBDA_GRADIENT_PENALTY * ((disc_gradients.norm(2, dim=1) - 1) ** 2).mean()

        # Add all together
        disc_loss = disc_real_pred_mean + disc_fake_pred_mean + disc_gradient_penalty

    opt_disc.zero_grad()
    d_scaler.scale(disc_loss).backward()
    d_scaler.step(opt_disc)
    d_scaler.update()

    return disc_real_pred_mean, disc_fake_pred_mean, disc_loss, disc_gradient_penalty


def train_gen_cycle_wgan_gp(gen_D, gen_N, disc_N, disc_D, night, fake_night, day, fake_day, l1, opt_gen, g_scaler):
    with torch.cuda.amp.autocast(enabled=False):
        # Adversarial losses
        D_N_fake = disc_N(fake_night)
        D_D_fake = disc_D(fake_day)
        loss_G_N = -torch.mean(D_N_fake)
        loss_G_D = -torch.mean(D_D_fake)

        # Cycle losses
        cycle_day = gen_D(fake_night)
        cycle_night = gen_N(fake_day)
        cycle_day_loss = l1(day, cycle_day) * config.LAMBDA_CYCLE_W
        cycle_night_loss = l1(night, cycle_night) * config.LAMBDA_CYCLE_W

        # Add all together
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

    return loss_G_D, loss_G_N, cycle_day_loss, cycle_night_loss, G_loss


def compute_gradient_penalty(disc, real_images, fake_images):
    eps = torch.rand(1, requires_grad=True)
    eps = eps.expand(real_images.size())
    eps = eps.to(config.DEVICE)
    x_tilde = eps * real_images + (1 - eps) * fake_images.detach()
    x_tilde = x_tilde.to(config.DEVICE)
    pred_tilde = disc(x_tilde)
    return grad(outputs=pred_tilde, inputs=x_tilde,
                grad_outputs=torch.ones(pred_tilde.size()).to(config.DEVICE),
                create_graph=True, retain_graph=True, only_inputs=True)[0]


def train_cycle_wgan_gp(disc_N, disc_D, gen_D, gen_N, loader, opt_disc_N, opt_disc_D, opt_gen, l1, mse, d_scaler,
                        g_scaler, base_path, epoch, scheduler_opt_disc_N, scheduler_opt_disc_D, scheduler_opt_gen,
                        ciconv_D, ciconv_N, use_ciconv_d, use_ciconv_g, train_output_path_tail):
    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        if config.USE_ARCHITECTURE_B:
            day = ciconv_D(day)
            night = ciconv_N(night)

        with torch.cuda.amp.autocast(enabled=False):
            fake_night = gen_N(day)
            fake_day = gen_D(night)

        # Train Discriminators
        D_N_real_pred, D_N_fake_pred, D_N_loss, D_N_gradient_penalty = \
            train_disc_cycle_wgan_gp(disc_N, night, fake_night, opt_disc_N, d_scaler)

        D_D_real_pred, D_D_fake_pred, D_D_loss, D_D_gradient_penalty = \
            train_disc_cycle_wgan_gp(disc_D, day, fake_day, opt_disc_D, d_scaler)

        # Train Generators (once every x batches)
        gen_is_trained = False
        if idx % 5 == 0:
            G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss, G_loss = \
                train_gen_cycle_wgan_gp(gen_D, gen_N, disc_N, disc_D, night, fake_night, day, fake_day, l1, opt_gen,
                                        g_scaler)
            gen_is_trained = True

        # Save images
        if config.SAVE_MODEL:
            save_images(idx, loader, epoch, fake_day, fake_night, base_path, train_output_path_tail)

        # noinspection PyUnboundLocalVariable
        if config.LOG_TRAINING:
            log_training_statistics(
                use_ciconv_d, use_ciconv_g,
                # Generators and Discriminators
                gen_D, gen_N, disc_D, disc_N,
                # Discriminator losses
                D_D_real_pred, D_D_fake_pred, D_N_real_pred, D_N_fake_pred, D_D_loss, D_N_loss,
                # Generator losses
                G_D_loss=G_D_loss if gen_is_trained else None,
                G_N_loss=G_N_loss if gen_is_trained else None,
                G_D_cycle_loss=G_D_cycle_loss if gen_is_trained else None,
                G_N_cycle_loss=G_N_cycle_loss if gen_is_trained else None,
                G_loss=G_loss if gen_is_trained else None,
                # CycleWGAN-gp specific losses
                D_D_gradient_penalty=D_D_gradient_penalty, D_N_gradient_penalty=D_N_gradient_penalty
            )

        # Print progress
        if idx % math.ceil(len(loader) / 10) == 0:
            print(f"Batch {idx} out of {len(loader)} completed")

        if config.LEARNING_RATE_DECAY:
            scheduler_opt_disc_N.step()
            scheduler_opt_disc_D.step()
            scheduler_opt_gen.step()
