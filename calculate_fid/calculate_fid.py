import argparse
import os
import shutil
import sys

import generate_images

sys.path.insert(1, "../")
from utils import get_time, get_duration, dir_size


def main(dir_real, dir_fake, dir_to_convert=None, checkpoint_path=None, ciconv_gen=None, delete_fake_images=False):
    generate_image_dir = dir_to_convert and checkpoint_path and ciconv_gen is not None

    # Create empty directory if images are generated and given directory is not empty
    if generate_image_dir and dir_size(dir_fake) != 0:
        checkpoint_split = checkpoint_path.split('/')
        dir_fake_new = f"fake_images_{checkpoint_split[-1].split('.')[0]}_{checkpoint_split[-2]}"
        os.makedirs(dir_fake_new)
        dir_fake = dir_fake_new
        print(f"\n- Created fake directory: {dir_fake}")

    # Generate directory with fake images
    if generate_image_dir:
        time = get_time()
        print(f"\n- Image generation started...\n"
              f"dir_to_convert: {dir_to_convert} ({dir_size(dir_to_convert)}), dir_fake: {dir_fake},"
              f" checkpoint_path: {checkpoint_path}, ciconv_gen: {ciconv_gen}")
        generate_images.main(dir_to_convert, dir_fake, checkpoint_path, ciconv_gen)
        print(f"Image generation completed in {get_duration(get_time() - time)}")

    # Calculate FID score
    print(f"\n- FID score calculation started...\n"
          f"dir_real: {dir_real} ({dir_size(dir_real)}), dir_fake: {dir_fake} ({dir_size(dir_fake)})\n")
    time = get_time()
    os.system(f"python3 -m pytorch_fid {dir_real} {dir_fake}")
    print(f"FID score calculation completed in {get_duration(get_time() - time)}")

    # Delete generated images if not stated otherwise
    if generate_image_dir and delete_fake_images:
        if os.path.exists(dir_fake):
            shutil.rmtree(dir_fake)
            print(f"\n- Deleted fake directory: {dir_fake}")


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dir_real", type=str, help="File path to the directory with real (night) images.")
    parser.add_argument("--dir_fake", type=str,
                        help="File path to the directory with fake (night) images or file path to the parent "
                             "folder in which to create a new folder to store generated images.")
    parser.add_argument("--dir_to_convert", type=str,
                        help="File path to the directory with real images (day) to convert.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint")
    parser.add_argument("--delete_fake_images", action='store_true', help="True to delete generated images")

    # Parse arguments
    args = parser.parse_args()

    main(
        args.dir_real,
        args.dir_fake,
        args.dir_to_convert,
        args.checkpoint_path,
        ("gen_" in args.checkpoint_path) if args.checkpoint_path is not None else None,
        args.delete_fake_images
    )
