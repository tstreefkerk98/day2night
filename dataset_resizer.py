import random
import sys
import utils
from PIL import Image


def scale_image_dimensions(img_urls, scale):
    path = random.choice(img_urls)
    width, height = Image.open(path).size
    return tuple(int(dim / scale) for dim in (width, height))


def resize_dataset(scale):
    for day_part in ["day", "night"]:
        img_urls = utils.get_images(day_part, resize=True)
        dimension = scale_image_dimensions(img_urls, scale)
        for idx, img_url in enumerate(img_urls):
            img = Image.open(img_url)
            img = img.resize(dimension)
            img.save(f"data/train/{day_part}/{day_part}_{idx}.png")


if __name__ == "__main__":
    scale = eval(sys.argv[1])
    assert isinstance(scale, (int, float))
    assert scale >= 1
    resize_dataset(scale)
