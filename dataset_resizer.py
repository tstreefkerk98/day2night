import os
import sys
import utils
from PIL import Image


def resize_dataset():
    # for day_part in ["day", "night"]:
    for day_part in ["day"]:
        img_urls = utils.get_images(day_part + "_labels", resize=True)
        for idx, img_url in enumerate(img_urls):
            img = Image.open(img_url)
            img = img.resize((img_size, img_size))
            file_name = os.path.basename(os.path.normpath(img_url))
            img.save(f"data/train_original/{day_part + '_labels_resized'}/{file_name}")


if __name__ == "__main__":
    img_size = eval(sys.argv[1])
    assert isinstance(img_size, int)
    resize_dataset()
