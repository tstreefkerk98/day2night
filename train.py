import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import config
import env
import wandb
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint, print_duration, get_time, get_date_time, save_images, prob, \
    dir_contains_checkpoint_files, log_training_statistics
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import Generator as Generator
from torch.autograd import grad


# Train CycleGAN Begin -------------------------------------------------------------------------------------------------
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
        # One sided label smoothing Discriminator Night
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
                    epoch):
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
        save_images(idx, loader, epoch, fake_day, fake_night, base_path, train_output_path_tail)

        # Log training statistics
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


# --------------------------------------------------------------------------------------------------- Train CycleGAN End

# Train CycleWGAN-GP Begin ---------------------------------------------------------------------------------------------
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
                        g_scaler, base_path, epoch):
    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

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
        save_images(idx, loader, epoch, fake_day, fake_night, base_path, train_output_path_tail)

        # noinspection PyUnboundLocalVariable
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


# ----------------------------------------------------------------------------------------------- Train CycleWGAN-GP End

def main():
    training_start_time = get_time()
    print(
        f"Training started at {get_date_time(training_start_time)}, "
        f"{'' if use_ciconv else 'not '}using CIConv\n"
        "with settings:\n"
        f"GENERATOR_CICONV: {use_ciconv_g}\n"
        f"DISCRIMINATOR_CICONV: {use_ciconv_d}\n"
        f"BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"LEARNING_RATE_G: {config.LEARNING_RATE_GEN}\n"
        f"LEARNING_RATE_D: {config.LEARNING_RATE_DISC}\n"
        f"NUM_WORKERS: {config.NUM_WORKERS}\n"
        f"NUM_EPOCHS: {config.NUM_EPOCHS}\n"
        f"SAVE_MODEL: {config.SAVE_MODEL}\n"
        f"LOAD_MODEL: {config.LOAD_MODEL}\n"
        f"LAMBDA_CYCLE: {config.LAMBDA_CYCLE_W if use_cycle_wgan else config.LAMBDA_CYCLE}"
    )
    if use_cycle_wgan:
        print(f"LAMBDA_GRADIENT_PENALTY: {config.LAMBDA_GRADIENT_PENALTY}\n")

    disc_N = Discriminator(in_channels=3, use_ciconv=use_ciconv_d, use_cycle_wgan=use_cycle_wgan).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3, use_ciconv=use_ciconv_d, use_cycle_wgan=use_cycle_wgan).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9, use_ciconv=use_ciconv_g).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9, use_ciconv=use_ciconv_g).to(config.DEVICE)

    # Initialise optimizers
    if use_cycle_wgan:
        opt_disc_N = optim.Adam(
            list(disc_N.parameters()),
            lr=config.LEARNING_RATE_DISC,
            betas=(0.5, 0.999),
        )
        opt_disc_D = optim.Adam(
            list(disc_D.parameters()),
            lr=config.LEARNING_RATE_DISC,
            betas=(0.5, 0.999),
        )
    else:
        opt_disc = optim.Adam(
            list(disc_N.parameters()) + list(disc_D.parameters()),
            lr=config.LEARNING_RATE_DISC,
            betas=(0.5, 0.999),
        )
    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999),
    )

    # Error functions
    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    base_path = "ciconv/" if use_ciconv else "no_ciconv/"
    checkpoint_files = [train_output_path_tail + "_" + config.CHECKPOINT_GEN_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_GEN_D,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_D]
    models = [gen_N, gen_D, disc_N, disc_D]
    # noinspection PyUnboundLocalVariable
    optimizers = [opt_gen, opt_gen, opt_disc_N, opt_disc_D] if use_cycle_wgan else \
        [opt_gen, opt_gen, opt_disc, opt_disc]
    learning_rates = [config.LEARNING_RATE_GEN, config.LEARNING_RATE_GEN, config.LEARNING_RATE_DISC,
                      config.LEARNING_RATE_DISC]

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

    # if not (use_ciconv and use_cycle_wgan):
    wandb.watch(
        [gen_D, gen_N, disc_N, disc_D],
        criterion=None, log="gradients", log_freq=math.ceil(len(loader) / 5), idx=None, log_graph=False
    )

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # Epoch start time
        time = get_time()
        progress = f"{epoch + 1}/{config.NUM_EPOCHS}"
        print(f"Epoch: {progress}, batches: {len(loader)}, start time: {get_date_time(time)}")

        # Train model
        if use_cycle_wgan:
            train_cycle_wgan_gp(disc_N, disc_D, gen_D, gen_N, loader, opt_disc_N, opt_disc_D, opt_gen, L1, mse,
                                d_scaler, g_scaler, base_path, epoch)
        else:
            train_cycle_gan(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,
                            base_path, epoch)

        # Print epoch duration
        print_duration(get_time() - time, "Epoch", progress)

        # Save model
        if config.SAVE_MODEL:
            for i in range(len(checkpoint_files)):
                save_checkpoint(models[i], optimizers[i], filename=base_path + "checkpoints/" + checkpoint_files[i],
                                epoch=epoch)

    # Print training time
    print_duration(get_time() - training_start_time, "Training", f"{config.NUM_EPOCHS} epochs")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cycle_wgan", action='store_true', help="True to use CycleWGAN-gp")
    parser.add_argument("--cycle_gan", action='store_true', help="True to use CycleGAN")
    parser.add_argument("--ciconv_disc", action='store_true', help="True to use CIConv in the Discriminators")
    parser.add_argument("--ciconv_gen", action='store_true', help="True to use CIConv in the Generators")
    parser.add_argument("--lr_gen", type=float, help="Generator learning rate, if not given a default will be used")
    parser.add_argument("--lr_disc", type=float,
                        help="Discriminator learning rate, if not given a default will be used")
    parser.add_argument("--batch_size", type=int, help="Batch size, if not given a default will be used")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs, if not given a default will be used")
    parser.add_argument("--file_tail", type=str,
                        help="File tail (used for storing files), if not given a default will be used")
    parser.add_argument("--l_cycle", type=int, help="Lambda cycle, if not given a default will be used")
    parser.add_argument("--l_gradient_pen", type=int,
                        help="Lambda gradient penalty, if not given a default will be used")
    parser.add_argument("--dont_save", action='store_true', help="Add to not save model")
    parser.add_argument("--dont_load", action='store_true', help="Add to not load model")

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments
    use_cycle_wgan = args.cycle_wgan
    assert args.cycle_gan ^ use_cycle_wgan
    use_ciconv_d = args.ciconv_disc
    use_ciconv_g = args.ciconv_gen
    assert not (use_ciconv_g and use_ciconv_d)
    use_ciconv = use_ciconv_d or use_ciconv_g
    if args.lr_disc is not None:
        config.LEARNING_RATE_DISC = args.lr_disc
    if args.lr_gen is not None:
        config.LEARNING_RATE_GEN = args.lr_gen
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_epochs is not None:
        config.NUM_EPOCHS = args.num_epochs
    train_output_path_tail = args.file_tail if args.file_tail is not None else "test"
    if args.l_cycle is not None:
        if use_cycle_wgan:
            config.LAMBDA_CYCLE_W = args.l_cycle
        else:
            config.LAMBDA_CYCLE = args.l_cycle
    if args.l_gradient_pen is not None:
        config.LAMBDA_GRADIENT_PENALTY = args.l_gradient_pen
    config.SAVE_MODEL = not args.dont_save
    config.LOAD_MODEL = not args.dont_load

    # Define wandb config object
    config_obj = {
        "ciconv_d": use_ciconv_d,
        "ciconv_g": use_ciconv_g,
        "learning_rate_d": config.LEARNING_RATE_DISC,
        "learning_rate_g": config.LEARNING_RATE_GEN,
        "lambda_cycle": config.LAMBDA_CYCLE_W if use_cycle_wgan else config.LAMBDA_CYCLE,
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "file_extension": train_output_path_tail
    }
    if use_cycle_wgan:
        config_obj["lambda_gradient_penalty"] = config.LAMBDA_GRADIENT_PENALTY

    # Login and initialise wandb
    wandb.login(key=env.WANDB_KEY)
    wandb.init(
        project=("day2night-cycle-wgan" if use_cycle_wgan else "day2night-cycle-gan"),
        entity=env.WANDB_ENTITY,
        config=config_obj
    )

    main()
