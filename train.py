import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import config
import env
import wandb
from ciconv2d import CIConv2d
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint, print_duration, get_time, get_date_time, \
    dir_contains_checkpoint_files
from torch.utils.data import DataLoader
from discriminator_model import Discriminator
from generator_model import Generator as Generator
from train_wgan_gp import train_cycle_wgan_gp
from train_gan import train_cycle_gan


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
        f"LEARNING_RATE_DECAY: {config.LEARNING_RATE_DECAY}\n"
        f"NUM_WORKERS: {config.NUM_WORKERS}\n"
        f"NUM_EPOCHS: {config.NUM_EPOCHS}\n"
        f"USE_ARCHITECTURE_B: {config.USE_ARCHITECTURE_B}\n"
        f"SAVE_MODEL: {config.SAVE_MODEL}\n"
        f"LOAD_MODEL: {config.LOAD_MODEL}\n"
        f"LOG_TRAINING: {config.LOG_TRAINING}\n"
        f"CLAMP_W: {config.CLAMP_W}\n"
        f"LAMBDA_CYCLE: {config.LAMBDA_CYCLE_W if use_cycle_wgan else config.LAMBDA_CYCLE}"
    )
    if use_cycle_wgan:
        print(f"LAMBDA_GRADIENT_PENALTY: {config.LAMBDA_GRADIENT_PENALTY}\n")

    in_channels = 1 if config.USE_ARCHITECTURE_B else 3
    disc_N = Discriminator(
        in_channels=in_channels,
        use_ciconv=use_ciconv_d,
        use_cycle_wgan=use_cycle_wgan,
        clamp_W=config.CLAMP_W
    ).to(config.DEVICE)
    disc_D = Discriminator(
        in_channels=in_channels,
        use_ciconv=use_ciconv_d,
        use_cycle_wgan=use_cycle_wgan,
        clamp_W=config.CLAMP_W
    ).to(config.DEVICE)

    gen_D = Generator(
        img_channels=in_channels,
        num_residuals=9,
        use_ciconv=use_ciconv_g,
        clamp_W=config.CLAMP_W,
        out_channels=in_channels
    ).to(config.DEVICE)
    gen_N = Generator(
        img_channels=in_channels,
        num_residuals=9,
        use_ciconv=use_ciconv_g,
        clamp_W=config.CLAMP_W,
        out_channels=in_channels
    ).to(config.DEVICE)

    if config.USE_ARCHITECTURE_B:
        ciconv_D = CIConv2d('W', k=3, scale=0.0, clamp_W=config.CLAMP_W)
        ciconv_N = CIConv2d('W', k=3, scale=0.0, clamp_W=config.CLAMP_W)
    else:
        ciconv_D, ciconv_N = None, None

    # Initialise optimizers
    if config.LEARNING_RATE_DECAY:
        gamma_disc = (config.LEARNING_RATE_DECAY / config.LEARNING_RATE_DISC) ** (1 / config.NUM_EPOCHS)
        gamma_gen = (config.LEARNING_RATE_DECAY / config.LEARNING_RATE_GEN) ** (1 / config.NUM_EPOCHS)

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
        scheduler_opt_disc_N, scheduler_opt_disc_D = None, None
        if config.LEARNING_RATE_DECAY:
            # noinspection PyUnboundLocalVariable
            scheduler_opt_disc_N = optim.lr_scheduler.ExponentialLR(opt_disc_N, gamma=gamma_disc)
            scheduler_opt_disc_D = optim.lr_scheduler.ExponentialLR(opt_disc_D, gamma=gamma_disc)
    else:
        opt_disc = optim.Adam(
            list(disc_N.parameters()) + list(disc_D.parameters()),
            lr=config.LEARNING_RATE_DISC,
            betas=(0.5, 0.999),
        )
        scheduler_opt_disc = None
        if config.LEARNING_RATE_DECAY:
            # noinspection PyUnboundLocalVariable
            scheduler_opt_disc = optim.lr_scheduler.ExponentialLR(opt_disc, gamma=gamma_disc)
    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999),
    )
    scheduler_opt_gen = None
    if config.LEARNING_RATE_DECAY:
        # noinspection PyUnboundLocalVariable
        scheduler_opt_gen = optim.lr_scheduler.ExponentialLR(opt_gen, gamma=gamma_gen)

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

    if config.LOG_TRAINING:
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
            # noinspection PyUnboundLocalVariable
            train_cycle_wgan_gp(disc_N, disc_D, gen_D, gen_N, loader, opt_disc_N, opt_disc_D, opt_gen, L1, mse,
                                d_scaler, g_scaler, base_path, epoch, scheduler_opt_disc_N, scheduler_opt_disc_D,
                                scheduler_opt_gen, ciconv_D, ciconv_N, use_ciconv_d, use_ciconv_g,
                                train_output_path_tail)
        else:
            # noinspection PyUnboundLocalVariable
            train_cycle_gan(disc_N, disc_D, gen_D, gen_N, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler,
                            base_path, epoch, scheduler_opt_disc, scheduler_opt_gen, use_ciconv_d, use_ciconv_g,
                            train_output_path_tail)

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
    parser.add_argument("--lr_decay", type=float,
                        help="Value of final learning rates, if not given, no decay will occur")
    parser.add_argument("--file_tail", type=str,
                        help="File tail (used for storing files), if not given a default will be used")
    parser.add_argument("--train_dir", type=str, help="Path to the training data files.")
    parser.add_argument("--l_cycle", type=int, help="Lambda cycle, if not given a default will be used")
    parser.add_argument("--l_gradient_pen", type=int,
                        help="Lambda gradient penalty, if not given a default will be used")
    parser.add_argument("--use_arch_b", action='store_true', help="Add to use architecture B")
    parser.add_argument("--dont_save", action='store_true', help="Add to not save model")
    parser.add_argument("--dont_load", action='store_true', help="Add to not load model")
    parser.add_argument("--dont_log", action='store_true', help="Add to not log training statistics")
    parser.add_argument("--clamp_W", type=float, help="Clamp boundary for W, if not given W will not be clamped")

    # Parse arguments
    args = parser.parse_args()

    # Assign arguments
    use_cycle_wgan = args.cycle_wgan
    assert args.cycle_gan ^ use_cycle_wgan
    use_ciconv_d = args.ciconv_disc
    use_ciconv_g = args.ciconv_gen
    assert not (use_ciconv_g and use_ciconv_d)
    use_ciconv = use_ciconv_d or use_ciconv_g
    if args.lr_disc is not None: config.LEARNING_RATE_DISC = args.lr_disc
    if args.lr_gen is not None: config.LEARNING_RATE_GEN = args.lr_gen
    if args.batch_size is not None: config.BATCH_SIZE = args.batch_size
    if args.num_epochs is not None: config.NUM_EPOCHS = args.num_epochs
    if args.lr_decay is not None: config.LEARNING_RATE_DECAY = args.lr_decay
    if args.clamp_W is not None: config.CLAMP_W = args.clamp_W
    train_output_path_tail = args.file_tail if args.file_tail is not None else "test"
    if args.train_dir is not None: config.TRAIN_DIR = args.train_dir
    if args.l_cycle is not None:
        if use_cycle_wgan:
            config.LAMBDA_CYCLE_W = args.l_cycle
        else:
            config.LAMBDA_CYCLE = args.l_cycle
    if args.l_gradient_pen is not None: config.LAMBDA_GRADIENT_PENALTY = args.l_gradient_pen
    config.USE_ARCHITECTURE_B = args.use_arch_b
    config.SAVE_MODEL = not args.dont_save
    config.LOAD_MODEL = not args.dont_load
    config.LOG_TRAINING = not args.dont_log

    # Define wandb config object
    if config.LOG_TRAINING:
        config_obj = {
            "ciconv_d": use_ciconv_d,
            "ciconv_g": use_ciconv_g,
            "learning_rate_d": config.LEARNING_RATE_DISC,
            "learning_rate_g": config.LEARNING_RATE_GEN,
            "learning_rate_decay": config.LEARNING_RATE_DECAY,
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
