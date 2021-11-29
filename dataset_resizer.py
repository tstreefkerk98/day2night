import config
from dataset import DayNightDataset
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader


def resize_dataset(dataset, size=256):
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    resize = transforms.Compose([transforms.Resize((size, size))])
    for idx, (day, night) in enumerate(loader):
        day_res = resize(day)
        night_res = resize(night)
        save_image(day_res, f"data/day/day_{idx}.jpg")
        save_image(night_res, f"data/night/night_{idx}.jpg")


def main():
    dataset = DayNightDataset(
        root_day=config.TRAIN_DIR + "/day",
        root_night=config.TRAIN_DIR + "/night",
        transform=config.transforms
    )
    resize_dataset(dataset)


if __name__ == "__main__":
    main()
