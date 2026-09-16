import cv2
import json
import os, argparse

parser = argparse.ArgumentParser(description="postprocessing script")
parser.add_argument("--experiment_path", "-e", type=str, required=True)
parser.add_argument("--scans_path", "-s", type=str, required=True)
parser.add_argument("--crop_left", "-l", type=int, required=True)
parser.add_argument("--crop_top", "-t", type=int, required=True)
parser.add_argument("--flip", "-f", action="store_true")

args = parser.parse_args()

# Resulting images will be safed in a new folder inside result_set_path called final_images
if not os.path.exists(os.path.join(args.experiment_path, "final_images/")):
    os.mkdir(os.path.join(args.experiment_path, "final_images/"))

with open(os.path.join(args.experiment_path, "stats.json"), "r") as file:
    json_file = json.load(file)
    for stats in json_file:
        file_name = stats["name"]
        base_name, _ = os.path.splitext(file_name)
        scan_name = base_name + ".tif"
        colorized_image_name = "color" + file_name
        scale = stats["scale"]
        l_offset = int(stats["left"])

        # Load scan, flip horizontally and convert to Lab color space
        scan_image_path = os.path.join(args.scans_path, scan_name)
        scan_image = cv2.imread(scan_image_path)
        if args.flip:
            scan_image = cv2.flip(scan_image, 1)
        lab_scan_image = cv2.cvtColor(scan_image, cv2.COLOR_BGR2Lab)

        # Load colorized image, scale to original size and convert to Lab color space
        colorized_image_path = os.path.join(
            args.experiment_path, "color/", colorized_image_name
        )
        colorized_image = cv2.imread(colorized_image_path)

        colorized_image = cv2.resize(colorized_image, None, fx=1 / scale, fy=1 / scale)

        lab_colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2Lab)

        # Extract the a and b channels from the colorized image
        colorized_lab_ab_channels = lab_colorized_image[:, :, 1:]

        # Replace the a and b channels from the scan with the corresponding ones from the colorized image
        lab_scan_image[
            args.crop_top : args.crop_top + lab_colorized_image.shape[0],
            args.crop_left
            + l_offset : args.crop_left
            + l_offset
            + lab_colorized_image.shape[1],
            1:,
        ] = colorized_lab_ab_channels

        # Convert the resulting Lab image back to BGR color space and half its resolution
        final_bgr_image = cv2.cvtColor(lab_scan_image, cv2.COLOR_Lab2BGR)
        if args.flip:
            final_bgr_image = cv2.flip(final_bgr_image, 1)

        # Save the final image
        final_image_path = os.path.join(
            args.experiment_path, "final_images/", file_name
        )
        cv2.imwrite(final_image_path, final_bgr_image)

        print("Final image: " + file_name + " saved successfully.")
