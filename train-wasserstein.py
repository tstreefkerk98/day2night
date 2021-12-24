import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
import sys
import utils
import env
import wandb
from PIL import Image
from dataset import DayNightDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from discriminator_model_wasserstein import Discriminator
from generator_model_wasserstein import Generator as Generator
from torch.autograd import grad


def train_disc(disc, real, fake, opt_disc, d_scaler):
    with torch.cuda.amp.autocast(enabled=False):
        # predictions
        disc_real_pred = -disc(real)
        disc_fake_pred = disc(fake.detach())

        disc_real_pred_mean = torch.mean(disc_real_pred)
        disc_fake_pred_mean = torch.mean(disc_fake_pred)

        # gradient penalty
        disc_gradients = compute_gradient_penalty(disc, real, fake)
        disc_gradient_penalty = config.LAMBDA_GRADIENT_PENALTY * ((disc_gradients.norm(2, dim=1) - 1) ** 2).mean()

        # add all together
        disc_loss = disc_real_pred_mean + disc_fake_pred_mean + disc_gradient_penalty

    opt_disc.zero_grad()
    d_scaler.scale(disc_loss).backward()
    # if use_ciconv_d:
    #     nn.utils.clip_grad_norm_(disc.ciconv.parameters(), 5)
    d_scaler.step(opt_disc)
    d_scaler.update()

    return disc_real_pred_mean, disc_fake_pred_mean, disc_loss, disc_gradient_penalty


def train_gen(gen_D, gen_N, disc_N, disc_D, night, fake_night, day, fake_day, l1, opt_gen, g_scaler):
    with torch.cuda.amp.autocast(enabled=False):
        # adversarial losses
        D_N_fake = disc_N(fake_night)
        D_D_fake = disc_D(fake_day)
        loss_G_N = -torch.mean(D_N_fake)
        loss_G_D = -torch.mean(D_D_fake)

        # cycle losses
        cycle_day = gen_D(fake_night)
        cycle_night = gen_N(fake_day)
        cycle_day_loss = l1(day, cycle_day) * config.LAMBDA_CYCLE_WASSERSTEIN
        cycle_night_loss = l1(night, cycle_night) * config.LAMBDA_CYCLE_WASSERSTEIN

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

    return loss_G_D, loss_G_N, cycle_day_loss, cycle_night_loss, G_loss


def train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc_N, opt_disc_D, opt_gen, l1, d_scaler, g_scaler,
             base_path, epoch):
    for idx, (day, night) in enumerate(loader):
        day = day.to(config.DEVICE)
        night = night.to(config.DEVICE)

        with torch.cuda.amp.autocast(enabled=False):
            fake_night = gen_N(day)
            fake_day = gen_D(night)

        # Train Discriminators
        D_N_real_pred, D_N_fake_pred, D_N_loss, D_N_gradient_penalty = \
            train_disc(disc_N, night, fake_night, opt_disc_N, d_scaler)

        D_D_real_pred, D_D_fake_pred, D_D_loss, D_D_gradient_penalty = \
            train_disc(disc_D, day, fake_day, opt_disc_D, d_scaler)

        # Train Generators (once every x batches)
        if idx % 5 == 0:
            G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss, G_loss = \
                train_gen(gen_D, gen_N, disc_N, disc_D, night, fake_night, day, fake_day, l1, opt_gen, g_scaler)

            G_D_loss_mean, G_N_loss_mean, G_N_cycle_loss_mean, G_D_cycle_loss_mean = \
                map(lambda x: x.mean().item(),
                    [G_D_loss, G_N_loss, G_D_cycle_loss, G_N_cycle_loss])

            log_obj = {
                "Generator day loss": G_D_loss_mean,
                "Generator night loss": G_N_loss_mean,
                "Generator day cycle loss": G_D_cycle_loss_mean,
                "Generator night cycle loss": G_N_cycle_loss_mean,
                "Generators total loss": G_loss
            }

        if idx % math.ceil(len(loader) / 5) == 0 or idx + 1 == len(loader):
            current_epoch = config.CURRENT_EPOCH + epoch
            save_image(fake_day * 0.5 + 0.5,
                       f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_day_{current_epoch}_{idx}.png")
            save_image(fake_night * 0.5 + 0.5,
                       f"{base_path}/saved_images_{base_path}/{train_output_path_tail}_night_{current_epoch}_{idx}.png")

        if "log_obj" not in locals():
            log_obj = {}

        log_obj.update({
            "Discriminator day real prediction": D_D_real_pred,
            "Discriminator day fake prediction": D_D_fake_pred,
            "Discriminator day gradient penalty": D_D_gradient_penalty,
            "Discriminator night real prediction": D_N_real_pred,
            "Discriminator night fake prediction": D_N_fake_pred,
            "Discriminator night gradient penalty": D_N_gradient_penalty,
            "Discriminator day loss": D_D_loss,
            "Discriminator night loss": D_N_loss,
            "Generator day last gradient": gen_D.last.weight.grad.mean().item(),
            "Generator night last gradient": gen_N.last.weight.grad.mean().item(),
            "Generator day last gradient abs": torch.abs(gen_D.last.weight.grad).mean().item(),
            "Generator night last gradient abs": torch.abs(gen_N.last.weight.grad).mean().item(),
        })

        if use_ciconv_d:
            log_obj["Discriminator night CIConv scale"] = disc_N.ciconv.scale.item()
            log_obj["Discriminator day CIConv scale"] = disc_D.ciconv.scale.item()

        if use_ciconv_g:
            log_obj["Generator night CIConv scale"] = gen_N.ciconv.scale.item()
            log_obj["Generator day CIConv scale"] = gen_D.ciconv.scale.item()

        wandb.log(log_obj)
        # wandb.log({
        #     "Real_day": get_wandb_img(day),
        #     "Fake_day": get_wandb_img(fake_day),
        #     "Real_night": get_wandb_img(night),
        #     "Fake_night": get_wandb_img(fake_night),
        # })

        if idx % 50 == 0:
            print(f"Batch {idx} out of {len(loader)} completed")


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


def get_wandb_img(tensor):
    grid = make_grid(tensor * 0.5 + 0.5)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    pil_img = Image.fromarray(ndarr)
    return wandb.Image(pil_img)


def main():
    training_start_time = utils.get_time()
    print(
        f"Training started at {utils.get_date_time(training_start_time)}, "
        f"{'' if use_ciconv else 'not '}using CIConv\n"
        "with settings:\n"
        f"BATCH_SIZE: {config.BATCH_SIZE}\n"
        f"LEARNING_RATE_G: {config.LEARNING_RATE_GEN}\n"
        f"LEARNING_RATE_D: {config.LEARNING_RATE_DISC}\n"
        f"LAMBDA_CYCLE: {config.LAMBDA_CYCLE_WASSERSTEIN}\n"
        f"LAMBDA_GRADIENT_PENALTY: {config.LAMBDA_GRADIENT_PENALTY}\n"
        f"NUM_WORKERS: {config.NUM_WORKERS}\n"
        f"NUM_EPOCHS: {config.NUM_EPOCHS}\n"
        f"SAVE_MODEL: {config.SAVE_MODEL}\n"
        f"LOAD_MODEL: {config.LOAD_MODEL}\n"
        f"GENERATOR_CICONV: {use_ciconv_g}\n"
        f"DISCRIMINATOR_CICONV: {use_ciconv_d}\n"
    )

    # Initialise Discriminators and Generators
    disc_N = Discriminator(in_channels=3, use_ciconv=use_ciconv_d).to(config.DEVICE)
    disc_D = Discriminator(in_channels=3, use_ciconv=use_ciconv_d).to(config.DEVICE)
    gen_D = Generator(img_channels=3, num_residuals=9, use_ciconv=use_ciconv_g).to(config.DEVICE)
    gen_N = Generator(img_channels=3, num_residuals=9, use_ciconv=use_ciconv_g).to(config.DEVICE)

    # Initialise optimizers
    opt_disc_N = optim.Adam(
        list(disc_N.parameters()),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.9),
    )
    opt_disc_D = optim.Adam(
        list(disc_D.parameters()),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.9),
    )
    opt_gen = optim.Adam(
        list(gen_D.parameters()) + list(gen_N.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.9),
    )

    # Generator cycle loss function
    L1 = nn.L1Loss()

    checkpoint_files = [train_output_path_tail + "_" + config.CHECKPOINT_GEN_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_GEN_D,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_N,
                        train_output_path_tail + "_" + config.CHECKPOINT_CRITIC_D]
    models = [gen_N, gen_D, disc_N, disc_D]
    optimizers = [opt_gen, opt_gen, opt_disc_N, opt_disc_D]
    learning_rates = [config.LEARNING_RATE_GEN, config.LEARNING_RATE_GEN,
                      config.LEARNING_RATE_DISC, config.LEARNING_RATE_DISC]

    # Load checkpoint
    base_path = "ciconv/" if use_ciconv else "no_ciconv/"
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

    if not use_ciconv:
        wandb.watch(
            [gen_D, gen_N, disc_N, disc_D],
            criterion=None, log="gradients", log_freq=math.ceil(len(loader) / 5), idx=None, log_graph=False
        )

    # Training loop
    for epoch in range(config.NUM_EPOCHS):
        # epoch start time
        time = utils.get_time()
        progress = f"{epoch + 1}/{config.NUM_EPOCHS}"
        print(f"Epoch: {progress}, batches: {len(loader)}, start time: {utils.get_date_time(time)}")

        # train model
        train_fn(disc_N, disc_D, gen_D, gen_N, loader, opt_disc_N, opt_disc_D, opt_gen, L1, d_scaler, g_scaler,
                 base_path, epoch)

        # print epoch duration
        utils.print_duration(utils.get_time() - time, "Epoch", progress)

        # save model
        if config.SAVE_MODEL:
            for i in range(len(checkpoint_files)):
                save_checkpoint(models[i], optimizers[i], epoch,
                                filename=base_path + "checkpoints/" + checkpoint_files[i])

    # Print training time
    utils.print_duration(utils.get_time() - training_start_time, "Training", f"{config.NUM_EPOCHS} epochs")


def dir_contains_checkpoint_files(base_path, checkpoint_files):
    return all([os.path.exists(base_path + "checkpoints/" + checkpoint_file) for checkpoint_file in checkpoint_files])


if __name__ == "__main__":
    disc_uses_ciconv_d = sys.argv[1]
    assert disc_uses_ciconv_d.lower() in ["true", "false"]

    disc_uses_ciconv_g = sys.argv[2]
    assert disc_uses_ciconv_g.lower() in ["true", "false"]

    learning_rate_g = eval(sys.argv[3])
    assert isinstance(learning_rate_g, float)
    config.LEARNING_RATE_GEN = learning_rate_g

    learning_rate_d = eval(sys.argv[4])
    assert isinstance(learning_rate_d, float)
    config.LEARNING_RATE_DISC = learning_rate_d

    batch_size = eval(sys.argv[5])
    assert isinstance(batch_size, int)
    config.BATCH_SIZE = batch_size

    num_epochs = eval(sys.argv[6])
    assert isinstance(num_epochs, int)
    config.NUM_EPOCHS = num_epochs

    train_output_path_tail = sys.argv[7]
    assert isinstance(train_output_path_tail, str)

    use_ciconv_d = eval(disc_uses_ciconv_d.title())
    use_ciconv_g = eval(disc_uses_ciconv_g.title())
    use_ciconv = use_ciconv_d or use_ciconv_g

    wandb.login(key=env.WANDB_KEY)
    wandb.init(
        project="day2night-wasserstein",
        entity="tstreefkerk",
        config={
            "ciconv_d": use_ciconv_d,
            "ciconv_g": use_ciconv_g,
            "learning_rate_d": config.LEARNING_RATE_DISC,
            "learning_rate_g": config.LEARNING_RATE_GEN,
            "lambda_cycle": config.LAMBDA_CYCLE_WASSERSTEIN,
            "lambda_gradient_penalty": config.LAMBDA_GRADIENT_PENALTY,
            "epochs": config.NUM_EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "file_extension": train_output_path_tail
        })

    main()
