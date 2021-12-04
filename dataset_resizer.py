import sys
import utils
from PIL import Image


def make_image_square(img):
    color = "black"
    width, height = img.size
    if width == height:
        return img
    elif width > height:
        result = Image.new(img.mode, (width, width), color)
        result.paste(img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(img.mode, (height, height), color)
        result.paste(img, ((height - width) // 2, 0))
        return result


def resize_dataset(img_size):
    for day_part in ["day", "night"]:
        img_urls = utils.get_images(day_part, resize=True)
        for idx, img_url in enumerate(img_urls):
            img = Image.open(img_url)
            # img = make_image_square(img)
            img = img.resize((img_size, img_size))
            img.save(f"data/train/{day_part}/{day_part}_{idx}.png")


if __name__ == "__main__":
    img_size = eval(sys.argv[1])
    assert isinstance(img_size, int)
    resize_dataset(img_size)
