import os, glob
from PIL import Image
import argparse
import numpy as np
import json

parser = argparse.ArgumentParser(description="preprocessing script")
parser.add_argument("--scans_path", "-s", type=str, required=True)
parser.add_argument("--output_path", "-o", type=str, required=True)
parser.add_argument("--crop_left", "-l", type=int, required=True)
parser.add_argument("--crop_top", "-t", type=int, required=True)
parser.add_argument("--flip", "-f", action="store_true")

args = parser.parse_args()


def process_inputs(input_files_org, input_dir):
    output_file_list = []
    images_dir = os.path.join(input_dir, "inputgrayIMGs/")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    for file in input_files_org:
        with Image.open(file) as img:
            width, height = img.size
            right = width - args.crop_left
            bottom = height - args.crop_top

            img = img.crop((args.crop_left, args.crop_top, right, bottom))

            np_img = np.asarray(img)
            if np_img.dtype == "uint16":
                np_img = (np_img >> 8).astype("uint8")
                img = Image.fromarray(np_img).convert("L")
            elif np_img.dtype == "uint8":
                img = img.convert("L")
            else:
                raise NotImplementedError(
                    "Images with depths other than 8-bit and 16-bit are not implemented"
                )

            if args.flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            filename = os.path.basename(file)
            name, _ = os.path.splitext(filename)
            new_name = name + ".png"
            img.save(os.path.join(images_dir, new_name))
            output_file_list.append(new_name)
    with open(os.path.join(input_dir, "test_images_list.json"), mode="w") as file:
        json.dump(output_file_list, file)


input_files_org = sorted(glob.glob(os.path.join(args.scans_path, "*")))
if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
process_inputs(input_files_org, args.output_path)
